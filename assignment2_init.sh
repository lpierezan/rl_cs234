# screen
# git clone https://github.com/lpierezan/rl_cs234.git
# cd rl_cs234
# chmod +x *.sh
# /opt/deeplearning/binaries/tensorflow/tensorflow_gpu-1.14.0-cp35-cp35m-linux_x86_64.whl
sudo apt-get -y install ffmpeg
cd assignment2
virtualenv .env --python=python3
source .env/bin/activate
pip install /opt/deeplearning/binaries/tensorflow/tensorflow_gpu-1.14.0-cp35-cp35m-linux_x86_64.whl
pip install -r requirements.txt
pip install gym[atari]

# ready to go
rm -r results/q5_train_atari_nature/
