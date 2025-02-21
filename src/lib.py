import torch
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import numpy as np
from typing import Tuple

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device="cuda")


def ensure_float(a: np.ndarray):
    if a.dtype == np.int16:
        return a.astype(np.float32, order="C") / 32768.0
    return a


def ensure_int(a: np.ndarray):
    return (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16).T


def embed(a: Tuple[int, np.ndarray]):
    sampling_rate, wav = a
    speaker = model.make_speaker_embedding(
        torch.from_numpy(ensure_float(wav)).unsqueeze(0), sampling_rate
    )
    return speaker


def synth(speaker: torch.Tensor, text: str):
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="ja")
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes).cpu()
    return (model.autoencoder.sampling_rate, ensure_int(wavs[0].float().cpu().numpy()))
