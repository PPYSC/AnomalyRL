# AnomalyRL

## Project Overview

This repository accompanies the paper **"Generating Anomalous Code from Models of Code"**.  

AnomalyRL is a reinforcement learning-based approach for automatically generating anomalous code.

## Table of Contents

- [AnomalyRL](#anomalyrl)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Step 1: Initialize Configuration](#step-1-initialize-configuration)
    - [Step 2: Train the Reinforcement Learning Agent](#step-2-train-the-reinforcement-learning-agent)
    - [Step 3: Generate Anomalous Code](#step-3-generate-anomalous-code)


## Installation

**1. Clone the Repository**
   
```bash
git clone https://github.com/PPYSC/AnomalyRL.git
cd AnomalyRL
```
   
**2. Install Pearl**  

AnomalyRL depends on [Pearl](https://github.com/facebookresearch/Pearl). Please follow the installation instructions provided in the Pearl repository to set it up properly.

## Usage

### Step 1: Initialize Configuration

**1. Set the Code Generation Model**  

Modify `rl/src/generator/node_mask_codet5p.py` to specify the type and path of the code generation model. The existing implementation defines a model based on [CodeT5+](https://github.com/salesforce/CodeT5).

**2. Configure Reinforcement Learning Training**  

Update `corpus_path` and `check_point_prefix` in `rl/src/train.py`.  
- `corpus_path`: Path to the file containing training data for reinforcement learning.
- `check_point_prefix`: Prefix for saving agent checkpoints.

**3. Configure Anomalous Code Generation**  

Update `corpus_path`, `checkpoint_path`, `output_path`, and `number_of_generated_programs` in `rl/src/run_generate.py`.  
- `corpus_path`: Path to the file containing seed programs used for code generation. 
- `checkpoint_path`: Path to the saved agent checkpoint.  
- `output_path`: Path to the file where the generated programs will be stored.
- `number_of_generated_programs`: Number of programs to be generated.

### Step 2: Train the Reinforcement Learning Agent

Run the following command to train the agent using the dataset specified in `corpus_path`:

```bash
cd rl/src
python train.py
```

The trained model will be saved at the location defined by `check_point_prefix`.

### Step 3: Generate Anomalous Code

Run the following command to generate code using the trained agent:

```bash
cd rl/src
python run_generate.py
```

The generated programs will be stored in the file specified by `output_path`.



 