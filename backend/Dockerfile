# Use a lightweight Python base image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common wget && \
    apt-get clean

# Manually install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Copy requirements and install dependencies
COPY requirements.txt ./requirements.txt

# Install PyTorch with MPS support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY ./data /app/data

COPY . .

EXPOSE 5002
CMD ["python3", "api.py"]
