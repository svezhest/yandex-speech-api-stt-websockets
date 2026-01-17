FROM ubuntu:22.04@sha256:104ae83764a5119017b8e8d6218fa0832b09df65aae7d5a6de29a85d813da2fb

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        portaudio19-dev \
        libportaudio2 \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy proto directory into /app/proto
COPY proto /app/proto

# Debug: verify proto directory exists
RUN ls -la /app/proto
RUN ls -la /app/proto/yandex/cloud/ai/stt/v3

# Copy requirements
COPY requirements.txt .

# Debug: contents of requirements.txt
RUN cat requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

# Ensure output directory exists
RUN mkdir -p /app/yandex/cloud/ai/stt/v3

# Generate Python modules from proto
RUN python3 -m grpc_tools.protoc \
    --proto_path=/app/proto \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/proto/yandex/cloud/ai/stt/v3/stt.proto

# Copy main application files
COPY main.py .
COPY .env .

# Ensure __init__.py files exist
RUN touch /app/yandex/__init__.py && touch /app/yandex/cloud/__init__.py && touch /app/yandex/cloud/ai/__init__.py && touch /app/yandex/cloud/ai/stt/__init__.py && touch /app/yandex/cloud/ai/stt/v3/__init__.py

# Set Python path
ENV PYTHONPATH=/app

CMD ["python3", "-u", "main.py"]
