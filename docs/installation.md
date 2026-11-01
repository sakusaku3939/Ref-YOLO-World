# 🛠️ Installation

Ref-YOLO-World is an MMYOLO / MMDetection project. The reference environment is
CUDA 11.8 with PyTorch 2.1.2.

## Docker (recommended)

```bash
git clone --recursive git@github.com:sakusaku3939/Ref-YOLO-World.git
cd Ref-YOLO-World
docker build -t ref-yolo-world -f docker/Dockerfile .

docker run -it --rm --name ref-yolo-world \
  --gpus all --shm-size=2g \
  -v "$(pwd)":/workspace \
  ref-yolo-world
```

All commands in the other documents are meant to be run inside the container at
`/workspace`.

The TNL2K protocol additionally needs Octave for the official MATLAB evaluation
script:

```bash
docker build -t ref-yolo-world:tnl2k -f docker/Dockerfile.tnl2k .
```

## Manual installation

```bash
git clone --recursive git@github.com:sakusaku3939/Ref-YOLO-World.git
cd Ref-YOLO-World

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install 'setuptools<71' openmim
pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html \
  --no-build-isolation
mim install mmdet==3.3.0 mmengine==0.10.6 mmyolo==0.6.0
pip install git+https://github.com/lvis-dataset/lvis-api

pip install --no-build-isolation ./third_party/mmyolo
pip install --no-build-isolation -e .
pip install -r requirements/basic_requirements.txt
```

ONNX export additionally needs `requirements/onnx_requirements.txt`. Core ML and
TFLite conversion also need `requirements/mobile_requirements.txt`; see
[onnx2tf](https://github.com/PINTO0309/onnx2tf) for its extra dependencies.
