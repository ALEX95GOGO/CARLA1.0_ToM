# Human AI symbolic fusion

## System Overview
![System Overview](asset/overview.png)

## Demo
<p align="center">
  <img src="asset/VideoProject3.gif" alt="Demo Video" width="100%" />
</p>

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


## Scenarios
It is usually ideal to have curated experiments in the form of scenarios parsed through [ScenarioRunner](https://carla-scenariorunner.readthedocs.io/en/latest/).

For this purpose, we created a handy script that should be robust to some of the quirks of the existing implementation. This script ([`run_experiment.py`](../ScenarioRunner/run_experiment.py)) will automatically start recording for you AFTER the new map has been loaded, with a unique filename, all on a single client instance, so that you don't need to worry about a faulty recording or overwriting existing files. 

With `scenario_runner` v0.9.13, you should have already set these environment variables:
```bash
# on bash (Linux)
export CARLA_ROOT=/PATH/TO/carla/
export SCENARIO_RUNNER_ROOT=/PATH/TO/scenario_runner/
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg                                           
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI

# on Windows x64 Visual C++ Toolset
set CARLA_ROOT=C:PATH\TO\carla\
set SCENARIO_RUNNER_ROOT=C:PATH\TO\scenario_runner\
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI
```
Then to run our demo examples, use this command:
```bash
# on bash (Linux)
cd $SCENARIO_RUNNER_ROOT # go to scenario-runner
./run_experiment.py --title "dreyevr_experiment" --route $SCENARIO_RUNNER_ROOT/srunner/data/routes_custom.xml $SCENARIO_RUNNER_ROOT/srunner/data/town05_scenarios.json 0

# on Windows x64 Visual C++ Toolset
cd %SCENARIO_RUNNER_ROOT% # go to scenario-runner
python run_experiment.py --title "dreyevr_experiment" --route %SCENARIO_RUNNER_ROOT%\srunner\data\routes_custom.xml %SCENARIO_RUNNER_ROOT%\srunner\data\town05_scenarios.json 0
```
Note that you can rename the experiment to anything with `--title "your_experiment"`, and the resulting recording file will include this title in its filename. 

![Sample Scenario](asset/Screenshot%202025-03-05%20122829.png)


# Recording/Replaying a scenario
## Motivations
It is often useful to record a scenario of an experiment in order to reenact it in post and perform further analysis. We had to slightly augment the recorder and replayer to respect our ego-vehicle being persistent in the world, but all other functionality is maintained. We additionally reenact the ego-sensor data (including HMD pose and orientation) so an experimenter could see what the participant was looking at on every tick. For the full explanation of the Carla recorder see their [documentation](https://carla.readthedocs.io/en/0.9.13/adv_recorder/). 

## Recording
The easiest way to start recording is with the handy `PythonAPI` scripts. 
```bash
cd $CARLA_ROOT/PythonAPI/examples/
# this starts a recording session with 10 autonomous vehicles
./start_recording.py -n 10 # change -n param to change number of vehicles (default 10)
...
^C # stop recording with a SIGINT (interrupt)
```
- Note that the recording halts if a new world is loaded, so if you want to change the world make sure to restart the recording after the world has loaded.
- The recording saves the default file as a `test1.log` file in the default saved directory of your project:
	- If recording from editor: `carla/Unreal/CarlaUE4/Saved/test1.log`
	- If recording from package: `${PACKAGE}/Unreal/Saved/test1.log` <!-- TODO: CHECK THIS -->
- The recording should have relatively minimal impact on the performance of the simulator, but this likely varies by machine. The experience should be essentially the same. 
- Note that the recorder saves everything in binary, so it the raw `test1.log` file is not human-readable. It is often nice to read it however, in that case use:
	- ```bash
		# saves output (stdout) to recorder.txt
		./show_recorder_file_info.py -a -f /PATH/TO/RECORDER-FILE > recorder.txt 
		```
  - With this `recorder.txt` file (which holds a human-readable dump of the entire recording log) you can parse this file into useful python data structures (numpy arrays/pandas dataframes) by using our [DReyeVR parser](https://github.com/harplab/dreyevr-parser).
## Replaying
Begin a replay session through the PythonAPI as follows:
```bash
# note this does not rely on being in VR mode or not. 
./start_replaying.py # this starts the replaying session
```
- Note that in the replaying mode, all user inputs will be ignored in favour of the replay inputs. However, you may still use the following level controls:
  1. **Toggle Play/Pause** - Is done by pressing `SpaceBar`
  2. **Advance** - Is done by holding `Alt` and pressing `Left` arrow (backwards) or `Right` arrow (forwards)
  3. **Change Playback Speed** - Is done by holding `Alt` and pressing `Up` arrow (increase) or `Down` arrow (decrease)
  4. **Restart** - Is done by holding `Alt` and pressing `BackSpace`
  6. **Possess Spectator** - Is done by pressing `1` (then use `WASDEQ+mouse` to fly around)
  7. **Re-possess Vehicle** - Is done by pressing `2`

To get accurate screenshots for every frame of the recording, see below with [synchronized replay frame capture](#synchronized-replay-frame-capture)

**NOTE** We use custom config files for global and vehicle parameters in the simulator (see [below](Usage.md#using-our-custom-config-file)) and we also store these parameters in the recording file so that we can verify they are the same as the replay. For instance, we will automatically compare the recording's parameters versus the live parameters when performing a replay. Then if we detect any differences, we will print these as warnings so you can be aware. For instance, if you recorded with a particular vehicle and replay the simulator with a different vehicle loaded, we will let you know that the replay may be inaccurate and you are in uncharted territory.

