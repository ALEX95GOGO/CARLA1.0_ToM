# Human AI symbolic fusion

## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
cd symbiotic_ai
conda create -n symbiotic_ai python=3.7
conda activate symbiotic_ai
pip3 install -r requirements.txt
```

## Download and setup CARLA 0.9.13
mkdir -p carla
cd carla

## Grab the simulator and Additional Maps packages
```Shell
wget https://tiny.carla.org/carla-0-9-13-linux -O CARLA_0.9.13.tar.gz
wget https://tiny.carla.org/additional-maps-0-9-13-linux -O AdditionalMaps_0.9.13.tar.gz
tar -xf CARLA_0.9.13.tar.gz
tar -xf AdditionalMaps_0.9.13.tar.gz
rm CARLA_0.9.13.tar.gz AdditionalMaps_0.9.13.tar.gz
cd ..
```

## Install the Python API
```Shell
easy_install carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
# (Alternatively) python3 -m pip install carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
```

