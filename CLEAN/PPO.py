


import numpy as np
import random


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

import dgl

from enviroment.ChemEnv import ChemEnv
from enviroment.utils import selfLoop
from models import init_weights_recursive, BaseLine, CriticSqueeze

device = None

class PPOTrainer:

    
    def __init__(self,
                 env: ChemEnv,
                 batch_size: int,
                 timesteps_per_batch: int,
                 clip: float,
                 a_lr: float,
                 c_lr: float,
                 n_updates_per_iteration: int,
                 max_timesteps_per_episode: int,
                 gamma: float,
                 actor: nn.Module,
                 writer):
        """PPO Initialization

        Args:
            env (ChemEnv): environment that adhers to OpenAI gym interface
            batch_size (int): though not in original paper batch size improved performance, size of batch
            timesteps_per_batch (int): how many steps to take before we train on the new data
            clip (float): how much to clip gradients
            a_lr (float): actor learning rate
            c_lr (float): critic learning rate
            n_updates_per_iteration (int): how many times we train through the experience before generating new ones
            max_timesteps_per_episode (int): the largest number of steps we can take in an episode
            gamma (float): discount factor
            actor (nn.Module): actor module
            writer ([type]): logger for tensorboard stuff
        """
        print("correcto")
        self.writer = writer
        
        # Initialize hyperparameters for training with PPO        
        self._init_hyperparameters(batch_size,timesteps_per_batch,clip,a_lr,c_lr,n_updates_per_iteration,max_timesteps_per_episode, gamma)

        # Extract environment information
        self.env = env
        input_dim = env.num_node_feats

         # Initialize actor and critic networks
        self.critic = CriticSqueeze(input_dim,300)
        self.actor = actor
        
        # Initialize optimizers for actor and critic
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.a_lr,eps=1e-5, weight_decay=.001)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.c_lr,eps=1e-5, weight_decay=.001)

        self.actor.apply(init_weights_recursive)
        self.critic.apply(init_weights_recursive)     
     
        self.batch_iter = 0
        
    def to_device(self):
        """put actor and critic onto gpu
        """
        
        self.actor.cuda()
        self.critic.cuda()
        

    def assignActor(self,new_actor):
        
        self.actor = new_actor
    
    def _init_hyperparameters(self,batch_size,timesteps_per_batch,clip,a_lr,c_lr,n_updates_per_iteration,max_timesteps_per_episode, gamma):
        
        self.batch_size = batch_size
        self.timesteps_per_batch = timesteps_per_batch
        self.clip = clip
        self.a_lr = a_lr
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.c_lr = c_lr
        self.n_updates_per_iteration = n_updates_per_iteration
        self.gamma = gamma
     
    

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        t_so_far = 0
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            print(t_so_far)
            
            
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  
            print(np.sum(batch_lens))             
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations


            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts, batch_rtgs)
            A_k = batch_rtgs.to(device) - V.detach()                                                              

            
            
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        
            # This is the loop where we update our network for some n times
            
            train_data_tuple = []
            for i in range(len(batch_obs)):
                single_tuple = (batch_obs[i],batch_acts[i],batch_rtgs[i],batch_log_probs[i],A_k[i])
                train_data_tuple.append(single_tuple)
    
            random.shuffle(train_data_tuple)
            batchlet_obs,batchlet_acts,batchlet_rtgs,batchlet_log_probs,A_k_let = zip(*train_data_tuple)
                
            failed_outer = False
            i = 0

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                random.shuffle(train_data_tuple)
                i = 0
                batchlet_obs,batchlet_acts,batchlet_rtgs,batchlet_log_probs,A_k_let = zip(*train_data_tuple)
                
                failed_outer = False
            
                while i < len(batch_obs)-65:
                    
                    #get batches
                    batchlet_obs_slice = batchlet_obs[i : (i+self.batch_size)]
                    batchlet_acts_slice = torch.stack(batchlet_acts[i : (i+self.batch_size)],0).to(device)
                    batchlet_rtgs_slice = torch.stack(batchlet_rtgs[i : (i+self.batch_size)],0).to(device)
                    batchlet_log_probs_slice = torch.stack(batchlet_log_probs[i : (i+self.batch_size)],0).to(device)
                    batchlet_A_k_slice = torch.stack(A_k_let[i : (i+self.batch_size)],0).to(device)
                        
                        
                    #failed if KL is too high     
                    failed = self.train_on_batch(batchlet_obs_slice, batchlet_acts_slice, batchlet_rtgs_slice,
                                        batchlet_log_probs_slice,batchlet_A_k_slice)
                    if failed:
                        failed_outer = True
                        break
                    
                    i += self.batch_size
                
                if failed_outer:
                    break
                
    def train_on_batch(self, batch_obs, batch_acts, batch_rtgs, batch_log_probs, A_k):
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)
        failed = False
        
        
        
        kl_approx = torch.mean(batch_log_probs - curr_log_probs)
        self.writer.add_scalar('Approximate KL', kl_approx, self.batch_iter)

        if kl_approx>.06:
            failed=True
        
        ratios = torch.exp(curr_log_probs - batch_log_probs)

        # Calculate surrogate losses.
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k



        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)

        self.writer.add_scalar('Actor Loss', actor_loss, self.batch_iter)
        self.writer.add_scalar('Critic Loss', actor_loss, self.batch_iter)
        self.batch_iter += 1
#               
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)  
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)        
        self.actor_optim.step()
    
        return failed
    
    
    def generate_graphs(self, num_graphs):
        graph_list = []
        for i in range(num_graphs):
            obs = self.env.reset()
            for ep_t in range(self.max_timesteps_per_episode):
                action, log_prob = self.get_action(obs)
                obs, rew, done, reward_dict  = self.env.step(action[0])
                if done:
                    graph_list.append(selfLoop(self.env.stateSpaceGraph))
                    break
        print('xx')
        
        return (graph_list)
    

    def inference(self,verbose=False):
        reward = 0
        obs = self.env.reset()
        for ep_t in range(self.max_timesteps_per_episode):

            action, log_prob = self.get_action(obs,True)
            print(action)
            obs, rew, done, reward_dict  = self.env.step(action[0],verbose)
            reward += rew
            if done:
                break
        
        return self.env.StateSpace
        
    def rollout(self):
        print("rolling out")
        """
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        #self.render(50)
        
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        
        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        
        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        self.batch_reward_plot = 0
        
        '''Plotting Variables'''
        total_reward=0
        total_disc_reward = 0
        total_hist_reward = 0
        num_episodes = 0
        
        
        while t < self.timesteps_per_batch:
            
            num_episodes += 1
            
            
            ep_rews = [] # rewards collected per episode

            obs = self.env.reset()
            done = False
            
            
            
            
            reward_for_episode = 0 
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                #self.env.render()
                
                final = False
                if ep_t == self.max_timesteps_per_episode-1: #final step of the generation 
                    final = True

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append((obs[0].clone(),obs[1],obs[2]))

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew, done, reward_dict  = self.env.step(action[0],final_step = final)
                reward_for_episode += rew
                
                total_disc_reward += reward_dict['model_reward']
                total_hist_reward += reward_dict['property_reward']
                
                
                total_reward += rew #track total rewards to get reward per step
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                # If the environment tells us the episode is terminated, break
                if done:
                    break
                
            self.batch_reward_plot += reward_for_episode
        
            #self.ep_rewards_over_time.append(reward_for_episode)
            # Track episodic lengths and rewards
            #self.ep_rewards_over_time.append(ep_rews)
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        
        #batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        
        '''Logging'''
        
        
        
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.

        print("roll out")
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num episodes per batch, num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, mask_on = False):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        
        #test_out = self.Batch_norm_edge1([obs[0]],[obs[1]],[obs[2]])[0]
        
        test_spin = self.actor(dgl.add_self_loop(dgl.remove_self_loop(obs[0])),torch.cat([obs[1]],0).to(device),torch.cat([obs[2]],dim = 0),mask = mask_on)
        test_dist = Categorical(test_spin)
        test_action = test_dist.sample()
        test_log_prob = test_dist.log_prob(test_action)
    

        return test_action.detach().cpu().numpy(), test_log_prob.detach()

    def evaluate(self, batch_obs, batch_acts, batch_rtgs):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        
        values = []
        log_prob_list = []
        
        
        
        batch_form_obs = [[],[],[]]
        for i in range(len(batch_obs)):
            batch_form_obs[0].append(selfLoop(batch_obs[i][0]))
            batch_form_obs[1].append(batch_obs[i][1].to(device))
            batch_form_obs[2].append(batch_obs[i][2].to(device))
        
        
        graph_batch = dgl.batch(batch_form_obs[0])

        
        
        CC = self.critic(graph_batch.to(device),torch.cat(batch_form_obs[1], 0).to(device),torch.cat(batch_form_obs[2], 0).to(device))
        A_new = self.actor(dgl.batch(batch_form_obs[0]),torch.cat(batch_form_obs[1], 0).to(device),torch.cat(batch_form_obs[2], 0).to(device))
        
        
        
        new_dist = Categorical(A_new)
        new_log_prob = new_dist.log_prob(batch_acts.to(device).squeeze())
        
        
        

        
        return CC.squeeze(), new_log_prob






