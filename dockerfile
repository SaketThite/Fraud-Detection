# Dockerfile
# Use a pre-built Hugging Face DLC (Deep Learning Container)
#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.35.2-gpu-py310-cu121
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04	
# Copy the inference script into the container
COPY Inference.py /opt/ml/code/Inference.py

# Set the environment variables for your model
ENV SAGEMAKER_PROGRAM=Inference.py