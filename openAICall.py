from openai import OpenAI
import os

client = OpenAI()

# Get working dir to help with paths 
image_path = os.path.join(os.getcwd(), "images", "output_image_003.png")
json_path = os.path.join(os.getcwd(), "images", "output_image_003.json")

# Open the image to client files 
with open(image_path, "rb") as f:
    uploaded_image = client.files.create(file=f, purpose="vision")
with open(json_path, "rb") as j: 
        uploaded_json = client.files.create(file=j, purpose="user_data")

response = client.responses.create(
    model="gpt-5.4",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Please triage the labeled patients in red using the US Military SALT protocol. Return the results in a json where you add a field for each id that identifies the SALT level."},
            {"type": "input_image", "file_id": uploaded_image.id},
            {"type": "input_file", "file_id": uploaded_json.id},
        ]
    }]
)

print(response.output_text)