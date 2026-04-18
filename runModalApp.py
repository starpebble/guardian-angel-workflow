import modal
import os 
import json

#Challenge is SAM3.1 does not work with transformers yet, accoridng to GPT

image_name = "image_001"
image_ending = ".png"
prompt = "Label the injured soliders"


def main():
    # Get full path relative to where script is run
    image_path = os.path.join(os.getcwd(), "images", image_name + image_ending)
    print("INPUT IMAGE PATH: ", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("MAKING MODEL")
    Sam3 = modal.Cls.from_name("sam3-infer", "Sam3")
    model = Sam3()

    result_bytes = model.infer.remote(image_bytes, prompt)
    
    output_path = os.path.join(os.getcwd(), "images", "output_" + image_name + image_ending)
    # Save image
    with open(output_path, "wb") as f:
        f.write(result_bytes["image_with_boxes"])

    # Save JSON
    json_output_path = os.path.join(os.getcwd(), "images", "output_" + image_name + ".json")
    with open(json_output_path, "w") as f:
        json.dump(result_bytes["json_data"], f, indent=2)
    print("SAVED OUTPUTS")



if __name__ == "__main__":
    main()