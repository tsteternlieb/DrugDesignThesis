import time, random, dgl, torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from ..Rewards.rewards import FinalRewardModule
from ..enviroment.ChemEnv import ChemEnv
from ..models import BaseLine, init_weights_recursive
from ..PPO import PPO_MAIN


device = None


class GraphDataLoader():
    def __init__(self,path,batch_size, length):
        self.path = path
        self.length = length
        
        self.batch_size = batch_size
        self.indicies = list(range(length))
        random.shuffle(self.indicies)
        
        self.curr_idx = 0
        
    def __get_exs(self,indices):
        graphs, graph_dict = dgl.load_graphs(self.path, indices)
        last_action = torch.stack([graph_dict['last_action'][idx] for idx in indices],dim = 0)
        last_atom_feat = torch.stack([graph_dict['last_atom_feats'][idx] for idx in indices], dim = 0)
        action = torch.stack([graph_dict['actions'][idx] for idx in indices], dim = 0)
        graphs = dgl.batch(graphs)
        return graphs.to(device), torch.unsqueeze(last_action,dim=1).to(device), last_atom_feat.to(device), action.to(device)
    
    def __next__(self):
        if self.curr_idx + self.batch_size > self.length:
            self.curr_idx = 0 
            random.shuffle(self.indicies)       
        batch_indices = self.indicies[self.curr_idx: self.curr_idx+self.batch_size]
        self.curr_idx += self.batch_size
    
        return self.__get_exs(batch_indices)
        
    
class SupervisedTrainingWrapper():
    def __init__(self, input_dim, batch_size, num_atom_types, writer, data_set_size, path = './graph_decomp/full_chunka'):
        #svt = SupervisedTrainingWrapper(54,5,15,None)
        self.path = path        
        self.policy = None #BaseLine(input_dim,300,17+1).cuda()
        self.optim = Adam(self.policy.parameters(), lr=3e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.chunk_idx = 0
        
        self.writer = writer
        self.n_iter = 0
        self.cv_iter = 0
        
        self.batch_size = batch_size
        self.data_set_size = data_set_size
        self.dataIter = GraphDataLoader(path,batch_size,data_set_size)
        print(data_set_size)

    def CalcAccuracy(self):
        a,b,c,d = next(self.dataIter)
        return self.calc_accuracy(a,b,c,d)
    
    
    def calc_accuracy(self,graphs, last_action_nodes, last_atom_feats, actions):
        y_logits = self.policy(graphs, last_action_nodes, last_atom_feats)
        values, pred_labels = y_logits.max(dim=1)
        acc = sum(pred_labels == actions)/actions.size()[0]
        return acc
    
    def calc_accuracy_no_call(self,logits,actions):
        values, pred_labels = logits.max(dim=1)
        acc = sum(pred_labels == actions)/actions.size()[0]
        return acc
        
        
    def Train(self,num_epochs):
        steps_per_epoch = self.data_set_size//self.batch_size
        
        
        for i in range(num_epochs):
            acc = self.CalcAccuracy()
            t0 = time()
            for step in range(steps_per_epoch):
                self._train()
            self._train()
            t1 = time()
            
            print(f'Time for epoch {i} is {t1-t0}, random accuracy is {acc}')
            
            
    def _train(self,calc_accuracy = True, update = True):
        graphs, last_action_nodes, last_atom_feats, actions = next(self.dataIter)
        
        pred = self.policy.forward(graphs, last_action_nodes,last_atom_feats, softmax=False)
        loss = self.loss_fn(pred,actions.long())
        
        if calc_accuracy:
            acc = self.calc_accuracy_no_call(pred,actions)
            self.writer.add_scalar("Accuracy", acc, self.n_iter)
        
        if update:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
        self.writer.add_scalar("pre_train_loss", loss.detach(), self.n_iter)
        self.n_iter += 1


class Supervised_Trainer(SupervisedTrainingWrapper):
    def __init__(self,policy_model, **kwargs):
        self.policy = policy_model
        super().__init__(**kwargs) 
        
        
    def TrainModel(self,total_epochs):
        
        self.Train(total_epochs)
        return self.policy
    
    
class SupervisedToReinforcement():
    def __init__(self,run_title, rewards_list, chem_env_kwargs,train_kwargs, PPO_kwargs, svw_kwargs):
        
        self.run_title = run_title
        self.writer = SummaryWriter(f'./tb_logs/{run_title}/{run_title}_logs')
        
        self.reward_module =  FinalRewardModule(self.writer,rewards_list)
        
        chem_env_kwargs['num_chunks'] = train_kwargs['num_chunks']
        chem_env_kwargs['RewardModule'] = self.reward_module
        chem_env_kwargs['writer'] = self.writer
        
        self.ChemEnv = ChemEnv(**chem_env_kwargs)
        
        
        input_dim = chem_env_kwargs['num_node_feats']
        
        #self.policy = Spin2(input_dim,300,chem_env_kwargs['num_atom_types']).cuda()
        self.policy = BaseLine(input_dim,800,chem_env_kwargs['num_atom_types']+1).cuda()
        self.policy.apply(init_weights_recursive)
        
        
        
        svw_kwargs['writer'] = self.writer
        svw_kwargs['input_dim'] = input_dim
        svw_kwargs['num_atom_types'] = chem_env_kwargs['num_atom_types']
        
        print(svw_kwargs)
        self.svw = Supervised_Trainer(self.policy, **svw_kwargs)
        
        PPO_kwargs['env'] = self.ChemEnv
        PPO_kwargs['actor'] = self.policy
        PPO_kwargs['writer'] = self.writer
        self.PPO = PPO_MAIN(**PPO_kwargs)
        self.PPO.to_device(device)
        
    
    def Train(self,total_epochs, batch_size, epochs_per_chunk, num_chunks, PPO_steps, cv_path):
        
        self.svw.TrainModel(total_epochs)
    
        # torch.save({
        #     'model_state_dict': self.policy.state_dict(),
        #     'optimizer_state_dict': self.svw.optim.state_dict()
        #     }, f'./{self.run_title}/SavedModel')
        
        print("fra")
        # self.PPO.learn(PPO_steps)
        