"""ParkVision web demo: pick a parking-lot photo, see which spaces are free.

Run locally:   .venv\\Scripts\\python app.py   (then open the link it prints)
"""
import json

import gradio as gr
import torch
from PIL import ImageDraw

from model import crop_spaces, load_model, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("small_cnn_best.pt", device)

# Each camera always looks at the same spaces, so we only need one list of boxes per camera.
with open("layouts.json") as f:
    LAYOUTS = json.load(f)

CAMERA_NOTES = {
    "PUCPR": "PUCPR: a lot the model **never saw** during training (the honest test).",
    "UFPR04": "UFPR04: one of the cameras the model **was trained on**.",
    "UFPR05": "UFPR05: one of the cameras the model **was trained on**.",
}

GREEN = (0, 220, 0)
RED = (230, 30, 30)


def analyze(image, camera):
    if image is None:
        return None, "Pick an example below or upload a photo."
    image = image.convert("RGB")
    if image.size != (1280, 720):
        return None, ("This demo only works on photos from the three PKLot cameras (1280×720), "
                      "because it needs to know where each parking space is.")

    boxes = LAYOUTS[camera]
    labels, confidence = predict(model, crop_spaces(image, boxes), device)

    out = image.copy()
    draw = ImageDraw.Draw(out)
    for (x, y, w, h), label in zip(boxes, labels):
        draw.rectangle([x, y, x + w, y + h], outline=GREEN if label == 0 else RED, width=3)

    free = (labels == 0).sum().item()
    unsure = (confidence < 0.8).sum().item()
    summary = (f"### {free} of {len(boxes)} spaces free\n"
               f"{CAMERA_NOTES[camera]}\n\n"
               f"Green = empty, red = occupied. The model was unsure (under 80% confident) about {unsure} space(s).")
    return out, summary


with gr.Blocks(title="ParkVision") as demo:
    gr.Markdown(
        "# ParkVision: find free parking spaces from a camera image\n"
        "A small CNN classifies every parking space as empty or occupied. "
        "It was trained on one parking lot (UFPR) and scores **94.1%** on a different lot it never saw (PUCPR). "
        "[Code and write-up on GitHub](https://github.com/abhasuru/parkvision)"
    )
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", label="Parking lot photo")
            camera = gr.Radio(list(LAYOUTS), value="PUCPR", label="Which camera took this photo?")
            button = gr.Button("Find free spaces", variant="primary")
        with gr.Column():
            result = gr.Image(label="Model's prediction")
            summary = gr.Markdown()

    gr.Examples(
        examples=[
            ["examples/pucpr_morning.jpg", "PUCPR"],
            ["examples/pucpr_afternoon.jpg", "PUCPR"],
            ["examples/pucpr_evening.jpg", "PUCPR"],
            ["examples/ufpr04.jpg", "UFPR04"],
            ["examples/ufpr05.jpg", "UFPR05"],
        ],
        inputs=[photo, camera],
        outputs=[result, summary],
        fn=analyze,
        run_on_click=True,
    )
    button.click(analyze, inputs=[photo, camera], outputs=[result, summary])

if __name__ == "__main__":
    demo.launch()
