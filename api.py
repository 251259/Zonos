# api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import torch
import torchaudio
import io
import base64

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes

app = FastAPI(title="Zonos TTS API")

# Load your Zonos model (ensure this path is correct for your Railway deployment)
MODEL_TYPE = "Zyphra/Zonos-v0.1-transformer" # Or "Zyphra/Zonos-v0.1-hybrid"
model = None  # Initialize model globally

def get_zonos_model():
    global model
    if model is None:
        print(f"Loading {MODEL_TYPE} model...")
        model = Zonos.from_pretrained(MODEL_TYPE, device="cuda") # Assuming CUDA is available on Railway
        model.requires_grad_(False).eval()
        print(f"{MODEL_TYPE} model loaded successfully!")
    return model

class TTSRequest(BaseModel):
    text: str
    language: str = "en-us"
    speaker_audio_path: Optional[str] = None # You can adjust this based on how you want to pass speaker audio
    emotion: Optional[list[float]] = None
    vqscore_8: Optional[list[float]] = None
    fmax: Optional[float] = 22050.0
    pitch_std: Optional[float] = 20.0
    speaking_rate: Optional[float] = 15.0
    dnsmos_ovrl: Optional[float] = 4.0
    speaker_noised: bool = False
    cfg_scale: float = 2.0
    min_p: float = 0.1
    seed: int = 420
    unconditional_keys: Optional[list[str]] = None

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/generate_tts/")
async def generate_tts(request: TTSRequest, zonos_model: Zonos = Depends(get_zonos_model)):
    """Generates TTS audio based on the provided text and parameters."""
    try:
        torch.manual_seed(request.seed)

        speaker_embedding = None
        if request.speaker_audio_path:
            # In a real deployment, you'd need to handle speaker audio file upload
            # and loading from a storage service (not directly from path in Railway's file system)
            raise HTTPException(status_code=400, detail="Speaker audio file upload not yet implemented in this example. You'd need to handle file uploads and storage in a real application.")
            # Example (placeholder - needs proper file upload handling and storage):
            # wav, sr = torchaudio.load(request.speaker_audio_path)
            # speaker_embedding = zonos_model.make_speaker_embedding(wav, sr)

        emotion_tensor = torch.tensor(request.emotion) if request.emotion else None
        vq_tensor = torch.tensor(request.vqscore_8) if request.vqscore_8 else None

        cond_dict = make_cond_dict(
            text=request.text,
            language=request.language,
            speaker=speaker_embedding,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=request.fmax,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            dnsmos_ovrl=request.dnsmos_ovrl,
            speaker_noised=request.speaker_noised,
            device="cuda", # Assuming CUDA is available
            unconditional_keys=request.unconditional_keys if request.unconditional_keys else set(),
        )
        conditioning = zonos_model.prepare_conditioning(cond_dict)

        codes = zonos_model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30, # Adjust as needed
            cfg_scale=request.cfg_scale,
            batch_size=1,
            sampling_params=dict(min_p=request.min_p),
            progress_bar=False # Disable progress bar in API context
        )

        wavs = zonos_model.autoencoder.decode(codes).cpu()
        wav_output = wavs[0] # Take the first (and only) batch item

        # Save to in-memory buffer
        output_buffer = io.BytesIO()
        torchaudio.save(output_buffer, wav_output, zonos_model.autoencoder.sampling_rate, format="wav")
        output_buffer.seek(0) # Reset buffer to the beginning

        # Encode to base64 for easy transmission in JSON
        audio_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "sampling_rate": zonos_model.autoencoder.sampling_rate,
            "seed": request.seed,
            "message": "TTS audio generated successfully!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Or your desired port
