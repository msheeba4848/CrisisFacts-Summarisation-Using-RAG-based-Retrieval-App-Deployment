# Use NVIDIA CUDA runtime image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Install dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-distutils python3.10-dev && \
    apt-get clean

# Manually install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy requirements
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY ./data /app/data

COPY . .

EXPOSE 5002
CMD ["python3", "api.py"]
