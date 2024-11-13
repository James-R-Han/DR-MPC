
## Setup Instructions

Assuming your current directory path is 'clone_location'.

1. **Create and activate the conda environment**  
   Ensure you have conda installed. Then, run the following commands to create and activate the environment:

   ```bash
   git clone https://github.come/James-R-Han/DR-MPC
   cd clone_location/DR-MPC
   conda env create -f environment.yml
   conda activate social_navigation
   ```
   Note: you may need to adjust the torch and cuda version to match your device

2. **Clone and install RVO2**  
   Next, clone the Python RVO2 repository and install it:

   Note: you should have cmake installed.
   ```
   sudo apt update
   sudo apt install cmake
   ```

   ```bash
   git clone https://github.com/sybrenstuvel/Python-RVO2.git
   cd Python-RVO2
   python setup.py build
   python setup.py install
   ```

3. **Clone and install pysteam**  
   Clone the pysteam repository and install it:

   ```bash
   cd clone_location/DR-MPC
   git clone https://github.com/utiasASRL/pysteam.git
   ```

4. **Setup PYTHONPATH**  
   Navigate to the `DR-MPC` directory and update your `PYTHONPATH`:

   ```bash
   cd clone_location/DR-MPC
   export PYTHONPATH="${PYTHONPATH}:clone_location/DR-MPC"
   ```

5. **Run the training script**  
   This command will train a policy specified in 'scripts/configs/config_general' for a number of trials on different seeds specified in 'scripts/configs/config_training'

   Note: for video generation you'll need ffmpeg
   ```bash
   sudo apt install ffmpeg
   ```

   ```bash
   cd clone_location/DR-MPC
   python3 scripts/online_continuous_task.py
   ```

6. **Comparing the results of different training runs**
   An example is already setup in scripts/compare_training_multirun.py

   ```bash
   python3 scripts/compare_training_multirun.py
   ```

The codebase is largely seprated into the environment ('environment') and training ('scripts'). The environment is modularized into the path tracking and human avoidance components as indicated by the folder names. RL learning algo (SAC), models (DRL, ResidualDRL, DR-MPC), OOD pipeline, etc. are all contained in the 'scripts' folder.