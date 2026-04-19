# Guardian Angel Workflow

Triages mascal pictures captured by drones or medics and sends them to the Guardian Angel console.

## Tooling

The workflow was made with OpenAI Codex.  Hugging Face gated access controls use of SAM3.  OpenAI APIs provide inference capabilities.

### Tech Stack

- SAM3
- OpenAI GPT5
- Python
- HuggingFace

### Setup

Used a python environemnt and downloaded necessary packages using pip. 

Using Modal to host the inference. Using facebook/Sam3. To run, need to enable hugging face (hf auth login) and set a modal secret token 
(also make sure to allow in the token settings for the models to read":

hf auth login 

modal setup - will make an api key and connect

"modal secret create hf-secret HF_TOKEN= [token]" - connects to hugging face for SAM

modal deploy [app name]

Then run. 

### Repos

Guardian Angel Components:

1. https://github.com/starpebble/guardian-angel-console
2. https://github.com/starpebble/guardian-angel-radio-app
3. https://github.com/starpebble/guardian-angel-workflow
