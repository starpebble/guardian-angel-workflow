from pathlib import Path
from sam3_ultra_modal import app, segment_image
import os


# Get the directory where this script lives
current_dir = Path(__file__).resolve().parent
# Change working directory to it
os.chdir(current_dir)
print("Working directory set to:", current_dir)

def run():
    image_path = Path(os.path.join(os.getcwd(), "images/image_001.png"))
    prompt = "identify the people person"

    image_bytes = image_path.read_bytes()

    with app.run():
        result_bytes = segment_image.remote(image_bytes, prompt)

    Path("output.png").write_bytes(result_bytes)
    print("Saved output.png")

if __name__ == "__main__":
    run()