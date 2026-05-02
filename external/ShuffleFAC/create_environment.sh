conda create -y -n deepship python==3.8.10
conda activate deepship

sudo apt update
pip install --upgrade pip

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install thop
pip install codecarbon==2.1.4
pip install -r requirements.txt