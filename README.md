# Trees-on-Farm-Plots
Identify trees on farm plots with MMSegmentation

conda create --name openmmlab python=3.8 -y
conda activate openmmlab

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .

pip install -r requirements.txt

python tools/train.py configs/segformer/segformer_mit-b5_8xb2-160k_talhoes-512x512.py