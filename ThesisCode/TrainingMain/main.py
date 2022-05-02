import yaml
from make_session import make, make2
import wandb


def main():
    
    # with open('./config_sweep.yaml') as file:
    #     config = yaml.safe_load(file)

    # with wandb.init(
    #     project="DrugNet",
    #     name='test',
    #     config=config,
    #     reinit=True
    # ):
    #     session = make2(wandb.config)
    #     session.Train()
    
    
    
    
    
    with open('./config_small.yaml') as file:
        config = yaml.safe_load(file)
    with wandb.init(
        project="DrugNet",
        name=config['RUN_TITLE'],
        config=config,
        reinit=True
    ): 
        run_title = config['RUN_TITLE'] 
        print(f'starting run with title: {run_title} ')
        session = make(config)[0]
        session.Train()
    

if __name__ == '__main__':
    main()
