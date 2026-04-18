import io
import os
import modal
import numpy as np 

# Setup modal app
app = modal.App("sam3-image-segmentation")

# Some decorator for the app - contianer to ru 
imag_container = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "huggingface_hub",
        "transformers>=4.58.0"
    )
)

# This is connected with the function below it
@app.function(
    image = imag_container,
    gpu="A10G",
    secrets=[modal.Secret.from_name("hf-secret")], #access token that i created on CLI
    timeout=900,
)
# Note if this is called with .remote, it goes to the cloud not local
def segment_image(image_bytes: bytes, text_prompt: str):
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    hf_token = os.environ["HF_TOKEN"]
    model_id = "facebook/sam3"

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    print("Setting up processor") 
    processor = Sam3Processor.from_pretrained(model_id, token=hf_token) # breaking here
    model = Sam3Model.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16,
    ).to("cuda")

    inputs = processor(
        images=pil_image,
        text=text_prompt,
        return_tensors="pt",
    )

    inputs = {
        k: (v.to("cuda") if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_masks(
        outputs,
        inputs.get("original_sizes"),
        inputs.get("reshaped_input_sizes"),
    )

    # Convert masks to numpy (keep as arrays, NOT list)
    masks = results[0]["masks"].cpu().numpy()

    # Combine masks into one
    combined_mask = None
    for m in masks:
        combined_mask = m if combined_mask is None else (combined_mask | m)

    # Convert original image
    img_np = np.array(pil_image).astype(np.float32)

    # Apply red overlay
    overlay = img_np.copy()
    overlay[combined_mask] = (
        0.6 * overlay[combined_mask] + 0.4 * np.array([255, 0, 0])
    )

    # Convert back to image
    out_img = Image.fromarray(overlay.astype(np.uint8))

    # Save to bytes
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    return buf.getvalue()

@app.local_entrypoint()
def main(image_path: str, prompt: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = segment_image.remote(image_bytes, prompt) # this is an app function
    print(result["num_masks"])
    print(result["scores"])