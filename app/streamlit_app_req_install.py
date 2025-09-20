!pip install transformers torch spacy openai pandas matplotlib streamlit scispacy sacremoses groq pyngrok
!python -m spacy download en_core_web_sm

from huggingface_hub import login
from google.colab import userdata
import os

# Get the Hugging Face token from Colab secrets
hf_token = userdata.get('HF_TOKEN')

# Log in to Hugging Face Hub using the token
if hf_token:
    login(token=hf_token)
    print("Logged in to Hugging Face Hub using Colab secrets.")
else:
    print("HF_TOKEN not found in Colab secrets. Please add it.")
    # You can optionally add a fallback here, like prompting the user
    # login() # Uncomment this line to fall back to prompting if the secret is not found
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz