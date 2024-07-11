# Traffic Scenarios Generation


# Script Usage

This repository contains a script for running Traffic Scenarios Generation experiments with various configurations. Below are the command-line arguments and their descriptions.

## Command-Line Arguments

1. **num_case** (type: int)
    - Description: The case number for the experiment. You can find the different experiment cases in `NeuroExperiment.py`.
    - Example: `--num_case 1`

2. **experiment_name_suffix** (type: int)
    - Description: Suffix to add to the experiment name for identification.
    - Example: `--experiment_name_suffix 2`

3. **main_folder** (type: string)
    - Description: The folder path where experiment results will be saved.
    - Example: `--main_folder "experiments"`

4. **repeat** (type: int)
    - Description: The number of times to repeat the experiment.
    - Example: `--repeat 5`

5. **optimization** (type: yes/no)
    - Description: Whether to perform BO hyperparameters optimization.
    - Example: `--optimization yes`

6. **load_model** (type: yes/no)
    - Description: Whether to load a pre-trained model.
    - Example: `--load_model --load`

7. **train_models** (type: yes/no)
    - Description: Whether to train the model again.
    - Example: `--train_models yes`

## Example Usage

To run the script with specific arguments, use the following command format:

```sh
python script_name.py --num_case <num_case> --experiment_name_suffix <experiment_name_suffix> --main_folder <main_folder> --repeat <repeat> --optimization <optimization> --load_model <load_model> --train_models <train_models>
