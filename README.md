# Target Specific Drug Design with Deep RL
A policy gradient approach for discovering novel drug candiates with desirable QSAR properties. Specifically we use the model to explore around promising drug canidates for recapitulating native function in p53 y220-c. In this setting we have a set of sucessful drugs which we would like to expand/iterate on in a ML-guided manner.

 ## Training
 Training occurs in a two stage process, an initial self-supervised pre training step and a secondary fine tuning with reinforcement learning. 


## Usage
 
- ### Model Training 
    to train the model, first set hyper paramaters in the config.yaml file. Next run python --config path_to_config. Tensorboard logs will be generated in a file with the name of the run title.
- ### Self Supervised Data Generation
    For generating pretraining data sepcify a path to a csv of small molecules with a smile column.

    python decomp.py --path path_to_csv
    
- ### Config File
    - Misc. 
        - RUN_TITLE: Title of the run
        - NUM_NODE_FEATS: Dimension of node features, Corresponds to the mol featurizer
        - MOL_FEATURIZER: Function to featurize molecules for model
        - REWARD_MODULES: List of reward module names for Reinforcement Learning (usage below)

    - Model Variables
        - HIDDEN_DIM: Hidden Dimension of Model
        - NUM_ATOM_TYPES: Bad code to be fixed

    - Proximal Policy Optimization variables:
        - PPO_BATCH_SIZE: Batch size for training
        - TIMESTEPS_PER_ITERATION: Number of steps to collect before training
        - CLIP: how much to clip the gradients for PPO
        - A_LR: actor learning rate
        - C_LR: critic learning rate
        - NUM_UPDATED_PER_ITERATION: How many times to train on experience
        - MAX_TIMESTEPS_PER_EPISODE: how long an episode can run, loosly corresponds to desired size of final molecules
        - GAMMA: Discount factor 

    - SUPERVISED_TRAINING_VARIABLES:
        - SUPERVISED_BATCH_SIZE: Batch Size for supervised training
        - DATASET_SIZE: Bad code to be fixed
        - PATH: Path to data
    

    - FINAL_TRAINING_VARIABLES:    
        - SV_EPOCHS: Number of runs through dataset
        - PPO_STEPS: Total number of steps for PPO training
    
- ### Reward Module
    - There are a number of reward modules to use, namely synthesizability, size, binding affinity, etc... More can be added, for example toxicity and solubility predictions. These need to be implemented using the SingleReward abstract class in rewards.py and then registered in config_utils.py. Then just add their names to the list in the config.

    - The Binding Affinity reward module uses autodock vina for calculations

    - There is tooling for profiling new reward modules in reward_profiling.ipynb before using them in RL training


- ### Inference
    [currently implementing] Inference can be run in two ways...
