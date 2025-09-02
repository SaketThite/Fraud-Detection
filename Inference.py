# inference.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def model_fn(model_dir):
    """
    Load the model and tokenizer from the specified directory.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the merged model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(data, model):
    """
    Generate a response based on the input payload.
    The payload is expected to be a dictionary with an 'inputs' key.
    """
    # Get the input text and generation parameters from the payload
    inputs = data.get("inputs", None)
    
    # Optional parameters for generation
    parameters = data.get("parameters", {})
    
    if not inputs:
        raise ValueError("Missing 'inputs' in the payload.")
    
    tokenizer = model["tokenizer"]
    model = model["model"]
    
    # Format the prompt exactly like your training data
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {inputs}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    tokenized_inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_inputs,
            **parameters,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant's response from the full text
    assistant_response = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    
    return {"generated_text": assistant_response.strip()}

def input_fn(request_body, content_type):
    """
    Parses the incoming request payload.
    """
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Content type '{content_type}' not supported")
    
def output_fn(prediction, accept_type):
    """
    Formats the prediction into a JSON response.
    """
    if accept_type == "application/json":
        return json.dumps(prediction), accept_type
    raise ValueError(f"Accept type '{accept_type}' not supported")