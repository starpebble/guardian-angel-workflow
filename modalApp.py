import io
import modal
import os 

app = modal.App("sam3-infer") # Name the app we are making

# The container that will be made at Modal and the packages needed there.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "torchvision",
        "Pillow",
    )
)
'''
@app.cls(image=image, gpu="L40S", 
        secrets=[modal.Secret.from_name("hf-secret")],
        scaledown_window=300
)
class Sam3:
    @modal.enter()
    def load(self):
        from transformers import pipeline

        token = os.environ["HF_TOKEN"]
        self.pipe = pipeline(
            "mask-generation",
            model="facebook/sam3",
            token = token,
            device=0,
        )

    @modal.method()
    def infer(self, image_bytes: bytes) -> dict[str, Any]:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = self.pipe(img, points_per_batch=64)

        masks = out.get("masks", [])
        # Simple overlay (very basic)

        overlay = np.array(img).copy()

        for m in masks:
            mask = np.array(m)
            overlay[mask > 0] = [255, 0, 0]  # red overlay

        result_img = Image.fromarray(overlay)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")

        return buf.getvalue()
'''

# Make sure to pass secrets; the token to authenticate
@app.cls(
    image=image,
    gpu="L40S",
    secrets=[modal.Secret.from_name("hf-secret")],
    scaledown_window=300,
)
class Sam3:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.environ["HF_TOKEN"]

        self.model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)

    @modal.method()
    def infer(self, image_bytes: bytes, text_prompt: str = "person"):
        import torch
        from PIL import Image, ImageDraw

        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size

        # Run model with TEXT prompt
        inputs = self.processor(
            images=img,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"] # absolute coordintates of bounding boxes: x_min, y_min, x_max, y_max of bounding box
        scores = results["scores"]

        # Draw boxes
        draw = ImageDraw.Draw(img)
        objects = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.tolist()

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"{score:.2f}", fill="red")
            draw.text((x1, min(width , max(0, y1 - 15) + 15)), str(i), fill="red")

            objects.append({    
                "id": i,
                "score": float(score),
                "width": float(x2-x1), # getting dimensions from coverage of the box
                "height": float(y2-y1),
                "x": x1,
                "y": y1,
            })

        # Save image to bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return {
            "image_with_boxes": buf.getvalue(),
            "json_data": {
                "text_prompt": text_prompt,
                "image_size": {"width": width, "height": height},
                "objects": objects,
            }
        }
    

@app.local_entrypoint()
def main(image_path: str, prompt: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    model = Sam3()
    result = model.infer.remote(image_bytes, prompt)
    print(result)