---
layout: default
title: Traffic Scenarios Generation
---

<h1 style="color:#1d66ab;">Traffic Scenarios Generation</h1>

Welcome to the project page!  
This site documents how to run experiments for generating traffic scenarios.

---
<p align="center" style="color:#c08c40; font-weight:bold;">
🚧 Page Under Construction – More content coming soon! 🚧
</p>
---

## Script Usage

This repository contains a script for running traffic scenario generation experiments with several configurable parameters.

## Command-Line Arguments

1. **num_case** (type: int)
    - Description: The case number for the experiment. You can find the different experiment cases in `NeuroExperiment.py`.
    - Example: `--num_case 1`

2. **experiment_name_suffix** (type: string)
    - Description: Suffix to add to the experiment name for identification. When performing multiple repetitions of an experiment, this string will indicate the folder suffix, followed by the seed number used .
    - Example: `--experiment_name_suffix METR16_experiment`

3. **main_folder** (type: string)
    - Description: The folder path where experiment results will be saved.
    - Example: `--main_folder "experiments"`

4. **repeat** (type: int)
    - Description: The number of times to repeat the experiment with several random seeds.
    - Example: `--repeat 5`

5. **optimization** (type: yes/no)
    - Description: Whether to perform BO hyperparameters optimization.
    - Example: `--optimization yes`

6. **load_model** (type: yes/no)
    - Description: Whether to load a pre-trained model.
    - Example: `--load_model yes`

7. **train_models** (type: yes/no)
    - Description: Whether to train the model again.
    - Example: `--train_models yes`

## Example Usage

To run the script with specific arguments, use the following command format:

```sh
python script_name.py --num_case <num_case> --experiment_name_suffix <experiment_name_suffix> --main_folder <main_folder> --repeat <repeat> --optimization <optimization> --load_model <load_model> --train_models <train_models>
```
Example
```sh
python3 test.py --neuroD --num_case 1 --experiment_name_suffix 2024_07_10_METR_16 --main_folder 2024_07_10_METR_16__OPT_split --repeation 5 --optimization yes --load_model no --train_models yes
```

## Help

To see the full list of available options, run:
```
python3 test.py --h
```
---
<h2 style="color:#c08c40;">Publications and Conferences</h2>

---
<h2 style="color:#c08c40;">Dataset, code and contacts</h2>


The code to run is available at:  
[https://github.com/mikeleikele/TransportScenariosGeneration](https://github.com/mikeleikele/TransportScenariosGeneration)

**Due to the large size of the dataset, it is not currently possible to host it on this page.**  
If you are interested in accessing the dataset, please contact us at  [michele.carbonera@unimib.it](mailto:michele.carbonera@unimib.it).

If you have any questions, please contact us at:  
[michele.carbonera@unimib.it](mailto:michele.carbonera@unimib.it)

**If you make use of this work, we kindly ask you to cite our related publications.**

---
<h2 style="color:#c08c40;">Acknowledgements</h2>

**This work has been funded by:**

<table>
  <tr>
    <td style="vertical-align: middle; width: 220px;">
      <img src="images/most-colore-412x291.webp" alt="MOST Logo" width="200" />
    </td>
    <td style="vertical-align: middle; padding-left: 15px;">
      <strong>MOST - National Sustainable Mobility Center</strong>, part of the European Union’s NextGenerationEU project (MOST - National Sustainable Mobility Center CN00000023, Italian Ministry of University and Research Decree No. 1033-17/06/2022, Spoke 8).
    </td>
  </tr>
</table>

