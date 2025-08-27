# CARLA1.0_ToM

## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
conda create -n symbiotic_ai python=3.7
conda activate symbiotic_ai
pip3 install -r requirements.txt
cd symbiotic_ai
python setup.py develop
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
