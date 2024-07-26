import numpy as np,torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,gym,random,math
from typing import NamedTuple,List,Tuple
from torch.distributions import Categorical
from collections import deque,namedtuple


Exp=namedtuple('Exp','s a r ns d i')

class PER:
    def __init__(self,cap:int,α:float=0.6,β:float=0.4,β_increment:float=0.001,ε:float=1e-5):
        self.capacity,self.α,self.β,self.β_increment,self.ε=cap,α,β,β_increment,ε
        self.tree,self.data,self.size,self.max_priority=np.zeros(2*cap-1),np.zeros(cap,dtype=object),0,1.0
    def _propagate(self,idx:int,change:float):
        parent=(idx-1)//2;self.tree[parent]+=change
        if parent!=0:self._propagate(parent,change)
    def _retrieve(self,idx:int,s:float)->int:
        left,right=2*idx+1,2*idx+2
        if left>=len(self.tree):return idx
        return self._retrieve(left,s) if s<=self.tree[left] else self._retrieve(right,s-self.tree[left])
    def add(self,experience:Exp,error:float):
        idx=self.size+self.capacity-1;self.data[self.size]=experience
        self.size=min(self.size+1,self.capacity);self.update(idx,error)
    def update(self,idx:int,error:float):
        priority=(error+self.ε)**self.α;change=priority-self.tree[idx]
        self.tree[idx]=priority;self._propagate(idx,change)
        self.max_priority=max(self.max_priority,priority)
    def sample(self,batch_size:int)->Tuple[List[Exp],List[int],np.ndarray]:
        batch,idxs,priorities=[],[],np.empty((batch_size,),dtype=np.float32)
        segment=self.tree[0]/batch_size;self.β=min(1.,self.β+self.β_increment)
        for i in range(batch_size):
            a,b=segment*i,segment*(i+1);s=random.uniform(a,b);idx=self._retrieve(0,s)
            priorities[i]=self.tree[idx];idxs.append(idx);batch.append(self.data[idx-self.capacity+1])
        sampling_probabilities=priorities/self.tree[0]
        weights=(self.size*sampling_probabilities)**-self.β;weights/=weights.max()
        return batch,idxs,weights
    
class NoisyLinear(nn.Module):
    
    def __init__(self,in_features:int,out_features:int,std_init:float=0.5):
        super(NoisyLinear,self).__init__()
        self.in_features,self.out_features,self.std_init=in_features,out_features,std_init
        self.weight_mu=nn.Parameter(torch.FloatTensor(out_features,in_features))
        self.weight_sigma=nn.Parameter(torch.FloatTensor(out_features,in_features))
        self.register_buffer('weight_epsilon',torch.FloatTensor(out_features,in_features))
        self.bias_mu=nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma=nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon',torch.FloatTensor(out_features))
        self.reset_parameters();self.reset_noise()
    def reset_parameters(self):
        mu_range=1/math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range,mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range,mu_range)
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.bias_sigma.size(0)))
    def reset_noise(self):
        epsilon_in=self._scale_noise(self.in_features)
        epsilon_out=self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    def _scale_noise(self,size:int):
        x=torch.randn(size);return x.sign().mul(x.abs().sqrt())
    def forward(self,inp:torch.Tensor):
        return F.linear(inp,self.weight_mu+self.weight_sigma*self.weight_epsilon,
                        self.bias_mu+self.bias_sigma*self.bias_epsilon) if self.training else F.linear(inp,self.weight_mu,self.bias_mu)
        
class DuelingDQN(nn.Module):
    def __init__(self,state_dim:int,action_dim:int,hidden_dim:List[int]):
        super(DuelingDQN,self).__init__()
        self.feature=nn.Sequential(nn.Linear(state_dim,hidden_dim[0]),nn.ReLU(),
                                   NoisyLinear(hidden_dim[0],hidden_dim[1]),nn.ReLU())
        self.advantage=nn.Sequential(NoisyLinear(hidden_dim[1],hidden_dim[2]),nn.ReLU(),
                                     NoisyLinear(hidden_dim[2],action_dim))
        self.value=nn.Sequential(NoisyLinear(hidden_dim[1],hidden_dim[2]),nn.ReLU(),
                                 NoisyLinear(hidden_dim[2],1))
    def forward(self,x:torch.Tensor):
        feature=self.feature(x);advantage=self.advantage(feature);value=self.value(feature)
        return value+advantage-advantage.mean(dim=-1,keepdim=True)
    
class Rainbow:
    def __init__(self,state_dim:int,action_dim:int,hidden_dim:List[int],learning_rate:float=3e-4,
                 gamma:float=0.99,tau:float=5e-3,alpha:float=0.2,n_step:int=3):
        self.q=DuelingDQN(state_dim,action_dim,hidden_dim)
        self.target_q=DuelingDQN(state_dim,action_dim,hidden_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer=optim.Adam(self.q.parameters(),lr=learning_rate)
        self.gamma,self.tau,self.alpha,self.n_step,self.action_dim=gamma,tau,alpha,n_step,action_dim
    def act(self,state:np.ndarray,epsilon:float=0.)->int:
        return random.randint(0,self.action_dim-1) if random.random()<epsilon else self.q(torch.FloatTensor(state)).argmax().item()
    def learn(self,experiences:List[Exp],weights:np.ndarray):
        states,actions,rewards,next_states,dones,_=map(torch.tensor,zip(*experiences))
        q_values=self.q(states).gather(1,actions.unsqueeze(1)).squeeze(1)
        next_q_values=self.target_q(next_states).max(1)[0]
        expected_q_values=(rewards+self.gamma**self.n_step*next_q_values*(1-dones)).detach()
        loss=(torch.FloatTensor(weights)*F.mse_loss(q_values,expected_q_values,reduction='none')).mean()
        policy=Categorical(F.softmax(self.q(states),dim=1))
        loss-=self.alpha*policy.entropy().mean()
        self.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(),1.0)
        self.optimizer.step()
        for target_param,param in zip(self.target_q.parameters(),self.q.parameters()):
            target_param.data.copy_(self.tau*param.data+(1.0-self.tau)*target_param.data)
        return loss.item(),policy.entropy().mean().item(),((q_values-expected_q_values).abs().detach().numpy()+1e-6)
    
    
class Trainer:
    def __init__(self,env:gym.Env,agent:Rainbow,batch_size:int=64,num_episodes:int=1000,
                 steps_per_episode:int=1000,update_every:int=4,start_epsilon:float=1.0,
                 end_epsilon:float=0.01,epsilon_decay:float=0.995):
        self.env,self.agent,self.batch_size,self.num_episodes=env,agent,batch_size,num_episodes
        self.steps_per_episode,self.update_every=steps_per_episode,update_every
        self.epsilon,self.end_epsilon,self.epsilon_decay=start_epsilon,end_epsilon,epsilon_decay
        self.memory,self.n_step_buffer=PER(1_000_000),deque(maxlen=agent.n_step)
    def n_step_learn(self):
        if len(self.n_step_buffer)<self.agent.n_step:return
        reward=sum([e.r*self.agent.gamma**i for i,e in enumerate(self.n_step_buffer)])
        state,action,_,next_state,done=self.n_step_buffer[-1]
        self.memory.add(Exp(self.n_step_buffer[0].s,action,reward,next_state,done,1),reward)
    def train(self):
        for i_episode in range(self.num_episodes):
            state,done=self.env.reset(),False;episode_reward=0
            for t in range(self.steps_per_episode):
                action=self.agent.act(state,self.epsilon)
                next_state,reward,done,_=self.env.step(action)
                self.n_step_buffer.append(Exp(state,action,reward,next_state,done,1))
                self.n_step_learn();state=next_state;episode_reward+=reward
                if len(self.memory.data)>self.batch_size and t%self.update_every==0:
                    experiences,indices,weights=self.memory.sample(self.batch_size)
                    loss,entropy,priorities=self.agent.learn(experiences,weights)
                    for idx,priority in zip(indices,priorities):self.memory.update(idx,priority)
                if done:break
            self.epsilon=max(self.end_epsilon,self.epsilon*self.epsilon_decay)
            print(f"Episode {i_episode+1}/{self.num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")
    def evaluate(self,num_episodes:int=100)->float:
        return np.mean([sum(self.env.step(self.agent.act(s))[1] for s in [self.env.reset()] for _ in iter(lambda:self.env.step(self.agent.act(s))[2],True)) for _ in range(num_episodes)])
if __name__=="__main__":
    env=gym.make('CartPole-v1')
    agent=Rainbow(env.observation_space.shape[0],env.action_space.n,[64,64,32])
    trainer=Trainer(env,agent);trainer.train()
    print(f"Average Reward: {trainer.evaluate():.2f}")