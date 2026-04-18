import modal

app = modal.App("sam3-ultralytics")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "ultralytics",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
)
def segment_image(image_bytes: bytes, prompt: str) -> bytes:
    import io
    import tempfile
    import numpy as np
    from PIL import Image
    from ultralytics import SAM

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        image_path = tmp.name

    # Adjust checkpoint name if needed based on the exact Ultralytics-supported SAM 3 weights you use.
    model = SAM("sam3_b.pt")

    # Text-prompt inference
    results = model(image_path, texts=[prompt])

    img_np = np.array(pil_image).astype(np.float32)
    overlay = img_np.copy()

    combined_mask = None
    if results and len(results) > 0 and getattr(results[0], "masks", None) is not None:
        masks_data = results[0].masks.data.cpu().numpy()
        for m in masks_data:
            m_bool = m.astype(bool)
            combined_mask = m_bool if combined_mask is None else (combined_mask | m_bool)

    if combined_mask is not None:
        overlay[combined_mask] = (
            0.6 * overlay[combined_mask] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
        )

    out_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()