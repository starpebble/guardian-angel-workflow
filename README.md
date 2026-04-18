# Guardian Angel Workflow

### Tech Stack

- SAM3
- OpenAI GPT5

### SETUP

Used a python environemnt and downloaded necessary packages using pip. 

Using Modal to host the inference. Using facebook/Sam3. To run, need to enable hugging face (hf auth login) and set a modal secret token 
(also make sure to allow in the token settings for the models to read":

hf auth login 

modal setup - will make an api key and connect

"modal secret create hf-secret HF_TOKEN= [token]" - connects to hugging face for SAM

modal deploy [app name]

Then run. 
