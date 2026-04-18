from pathlib import Path
import modal
import os

#from modalTest import app, segment_image # from the other file

from sam3_modal import app, segment_image


# Get the directory where this script lives
current_dir = Path(__file__).resolve().parent
# Change working directory to it
os.chdir(current_dir)
print("Working directory set to:", current_dir)


def run():
    image_path = image_path = os.path.join(os.getcwd(), "images/image_001.png")
    prompt = "identify the injured people and segment them"

    image_Path = Path(image_path)
    image_bytes = image_Path.read_bytes()

    with app.run():
        print("entering app ...")
        result_bytes = segment_image.remote(image_bytes, prompt)
        with open("output.png", "wb") as f:
            f.write(result_bytes)

    print("Saved output.png")


if __name__ == "__main__":
    run()