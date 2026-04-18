import modal

app = modal.App("sam3-official")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "huggingface_hub",
        "hydra-core",
        "omegaconf",
        "iopath",
        "timm",
        "opencv-python",
        "einops",
        "pycocotools",
        "psutil",
        "git+https://github.com/facebookresearch/sam3.git",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=1800,
)
def segment_image(image_bytes: bytes, text_prompt: str):
    import io
    import os
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import login
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    login(token=os.environ["HF_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Force one dtype for the model
    model = build_sam3_image_model()
    model = model.to(device)
    model = model.half()
    
    for p in model.parameters():
        if p.dtype.is_floating_point:
            p.data = p.data.half()
    for b in model.buffers():
        if b.dtype.is_floating_point:
            b.data = b.data.half()

    model.eval()


    processor = Sam3Processor(model)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            state = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output["masks"]
    if len(masks) == 0:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()

    combined_mask = None
    for m in masks:
        m_np = np.array(m, dtype=bool)
        combined_mask = m_np if combined_mask is None else (combined_mask | m_np)

    img_np = np.array(pil_image).astype(np.float32)
    overlay = img_np.copy()
    overlay[combined_mask] = (
        0.6 * overlay[combined_mask] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    )

    out_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()