from lib import synth, embed

import gradio as gr
from typing import Tuple, Optional
import numpy as np


def update_embedding(audio: Optional[Tuple[int, np.ndarray]]):
    if audio is None:
        return None
    return embed(audio)


def synthesize_audio(text: str, speaker_embedding):
    if speaker_embedding is None:
        return None
    print(f"Creating: {text}")
    return synth(speaker_embedding, text)


with gr.Blocks() as demo:
    speaker_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload/Paste Speaker Audio", type="numpy")
        with gr.Column():
            text_input = gr.Textbox(label="Input text for synthesis")
            synth_button = gr.Button("Synthesize Audio")
            audio_output = gr.Audio(label="Synthesized Audio", type="numpy")
    audio_input.change(fn=update_embedding, inputs=audio_input, outputs=speaker_state)
    synth_button.click(
        fn=synthesize_audio,
        inputs=[text_input, speaker_state],
        outputs=audio_output,
        concurrency_limit=1,
    )

demo.launch(server_name="0.0.0.0", share=True)
