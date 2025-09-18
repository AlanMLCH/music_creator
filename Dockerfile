# CUDA-enabled PyTorch base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y fluidsynth fluid-soundfont-gm && --no-install-recommends \
python3 python3-pip python3-venv git ffmpeg && \
rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Install PyTorch with CUDA and common libs
RUN python3 -m pip install --upgrade pip && \
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY . .


CMD ["bash", "-lc", "python -m src.train --config config/config.yaml"]