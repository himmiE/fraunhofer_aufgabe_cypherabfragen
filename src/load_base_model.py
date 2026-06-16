# includes function to load the basemodel `HuggingFaceTB/SmolLM2-135M-Instruct` from https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_from_huggingface(checkpoint):
    # loads tokenizer and model from huggingface and returns them

    device = "cpu" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.save_pretrained("./../model/base_model")
    tokenizer.save_pretrained("./../model/base_model")


    return tokenizer, model

def load_model_from_file(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer, model

def load_model(name):
    path = "./../model/" + name
    if os.path.exists(path):
        tokenizer, model = load_model_from_file(path)
    elif name == "base-model":
        tokenizer, model = load_model_from_huggingface("HuggingFaceTB/SmolLM2-135M-Instruct")
    else:
        return None, None
    return tokenizer, model
