import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

# Get the SageMaker session and role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define the location of your S3 model artifacts and ECR container image
model_data = f"s3://YOUR_BUCKET_NAME/models/your-model-name/model.tar.gz"
image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPOSITORY_NAME}:latest"

# Create a SageMaker model object
huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    image_uri=image_uri,
)

# Deploy the model
# Choose an instance type with enough VRAM, e.g., 'ml.g5.xlarge' for 24GB VRAM
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="your-model-endpoint-name"
)

print(f"Endpoint deployed at: {predictor.endpoint_name}")