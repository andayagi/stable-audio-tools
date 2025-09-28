import json
import os

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download

def get_pretrained_model(name: str):
    
    # Get HuggingFace token from environment if available
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model', token=hf_token)

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model', token=hf_token)
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model', token=hf_token)

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    return model, model_config