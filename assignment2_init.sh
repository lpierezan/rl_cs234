# git clone https://github.com/lpierezan/rl_cs234.git
sudo apt-get -y install ffmpeg
cd rl_cs234/assignment2
virtualenv .env --python=python3
source .env/bin/activate
pip install -r requirements.txt
pip install gym[atari]
