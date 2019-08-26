# screen
# git clone https://github.com/lpierezan/rl_cs234.git
# cd rl_cs234
# chmod +x *.sh
# /opt/deeplearning/binaries/tensorflow/
sudo apt-get -y install ffmpeg
cd assignment2
virtualenv .env --python=python3
source .env/bin/activate
pip install -r requirements.txt
pip install gym[atari]
