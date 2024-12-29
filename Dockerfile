# Use AWS Lambda Python base image for ARM64
FROM public.ecr.aws/lambda/python:3.11-arm64

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]