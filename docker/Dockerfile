FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel   \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip wheel \
    && pip3 install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir \
        opencv-python==4.9.0.80 \
        supervision \
        setuptools \
        openmim \
        onnx \
        onnxsim

RUN pip3 install 'setuptools<71' \
    && pip3 install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html --no-build-isolation \
    && mim install mmdet==3.3.0 \
    && mim install mmengine==0.10.6 \
    && mim install mmyolo==0.6.0 \
    && pip3 install git+https://github.com/lvis-dataset/lvis-api \
    && pip3 install numpy==1.23.5

COPY . /workspace
WORKDIR /workspace

RUN pip3 install --no-build-isolation ./third_party/mmyolo \
    && pip3 install --no-build-isolation -e . \
    && pip3 install --force-reinstall "transformers==4.51.3" \
    && pip3 install albumentations \
    && pip3 install "numpy<2.0"
