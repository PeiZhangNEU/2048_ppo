# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/expand_disk/ShaoWenLU_hmwk/ppo_2048/')  
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# 几个全局函数
def get_mean_and_std(x):
    '''计算一组数据的均值和方差'''
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    return mean, std

def combined_shape(length, shape=None):
    '''
    把length和shape结合起来得到列表，用于buffer的存储数据的形状初始化.
    比如Discrete环境的 actdim=(), 10，()就是得到[10,]
    例如：10,[100,3] 得到 [10,100,3]
         10, 3 得到 [10,3]
         10, None 得到 [10,]
         return A If B else C代表，如果B，返回A，否则返回C
         return *列表是返回列表里面的值，例如return *[1,2,3] = 1,2,3
    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def conv_mlp(obs_dim, act_dim):
    '''
    对于atari环境，输入是图片的情况来说，需要用到卷积处理图片，方便起见，我这里的网络结构直接制定了
    obs_dim = [1, 16, 16]
    '''
    # 先建立一个卷及模型
    conv_model = nn.Sequential(
                nn.Conv2d(obs_dim[0], 32, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
            )
    # 再进行全连接, 全连接之前必须知道卷积的输出拉直之后是什么形状
    tmp = torch.zeros(1, * obs_dim)
    feature_size = conv_model(tmp).view(1, -1).size(1)

    # 再建立一个线性模型
    linear_model = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim)
            )

    return conv_model, linear_model, feature_size

def count_vars(module):
    '''
    返回一个模型所有的参数数量
    '''
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    '''
    magic from rllab for computing discounted cumulative sums of vectors.就是给rewards计算Gt
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,    
         x1 + discount * x2,                      
         x2]                                      

        Output_t = \sum_{l=0}^inf [(discount)^l]*Input_t+l

    list[::-1]会返回list的倒叙列表，-1是步长
    '''
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# Actor的基础类以及离散Actor和连续Actor类。
class Actor(nn.Module):
    '''
    创建actor
    类里面的方法出现前置下划线是，代表这个函数是该类私有的，只能在内部调用
    这个类没有 __init__(self, inputs) 所以是不可实例化的类，只是一个用来继承的模板
    '''
    def _distribution(self, obs):
        '''
        这个是提示目前这个函数还没写，是一种技巧，先需要有一个这个函数，另一个类继承过来的时候再写
        obs的维度是[N,obs_dim]，N可以是1，这是就是单个的obs
        如在连续空间中，actor将产生[N,act_dim]维度的mu
        然后利用生成的参数产生分布dist，格式是dist(loc:size=[N,act_dim],scale:size=[N,act_dim])
        dist分布其实就是 pi(.|s)给定s时的分布函数
        '''
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        '''
        计算 dist.log_prob(a)
        '''
        raise NotImplementedError

    def forward(self, obs, act=None):
        '''
        这个函数是为了计算目前的logpa，操作的是批量数据，批量数据仅仅在update的时候需要用到！
        只在upadate这一步计算loss时才需要用到
        带梯度
        产生给定状态的分布dist
        计算分布下，给定动作对应的log p(a)
        actor里面forward一般是只接收批量的数据，每一步的计算用上面的函数
        '''
        dist = self._distribution(obs)   # \pi(\cdot|s)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(dist, act)
        return dist, logp_a
    
class MLPCategoricalActor(Actor):
    '''
    继承Actor类，并修改一些基类的方法，产生离散的分布，就是PMF概率质量分布率，用于处理离散动作空间 Discrete
    可以实例化
    '''
    def __init__(self, obs_dim, act_dim):
        '''初始一个logits网络，可以直接输出各个动作对应的概率'''
        super().__init__()
        self.conv_net, self.logits_net, self.feature_size = conv_mlp(obs_dim, act_dim)

    def _distribution(self, obs):
        '''返回离散分布dist [N,act_dim]，每个分布中，每个动作就对应一个确切的概率'''
        conv_feature = self.conv_net(obs).view(obs.size(0), -1)   # [N, conv_features num]
        # 如果第一维度不是1，这样就错了，得道 conv_features 之后要拉/直
        conv_feature = conv_feature.view(-1, self.feature_size) # 拉直变成 linear层输入的形状！！
        logits = self.logits_net(conv_feature)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        '''输出形为[N,]的logprob'''
        return pi.log_prob(act) # 离散动作空间，输入act的维度是[N,]，因为选择出来的动作是act_dim里面的一个概率最大的动作,然后输出也是[N,]
                                # 比如倒立摆小车，离散动作空间维度为2，但是最后输出的动作是左或者右，只有1维，这是离散动作的特点！
                                # 输入1个动作，那就输出1个这个动作对应的概率！


# Critic 只有一个基础MLP类，不需要基础类，直接一个可以实例化的类就行了。
class MLPCritic(nn.Module):
    '''Critic的输出维度只有[N,1]，输入是状态'''
    def __init__(self, obs_dim):
        super().__init__()
        self.vconv_net, self.vlogits_net, self.vfeature_size = conv_mlp(obs_dim, 1)
    
    def forward(self, obs):
        conv_feature = self.vconv_net(obs).view(obs.size(0), -1)   # [N, conv_features num]
        # 如果第一维度不是1，这样就错了，得道 conv_features 之后要拉/直
        conv_feature = conv_feature.view(-1, self.vfeature_size) # 拉直变成 linear层输入的形状！！
        logits = self.vlogits_net(conv_feature)
        return torch.squeeze(logits, -1)  # 保证Critic输出的价值的维度也是 [N,]

# 把Actor和Critic合并成一类
class MLPActorCritic(nn.Module):
    '''
    创建一个默认参数的，可以调用的ActorCritic网络
    '''
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        
        obs_dim = observation_space.shape

        act_dim = action_space.n
        self.pi = MLPCategoricalActor(obs_dim, act_dim)
        
        # 建立Critic策略v
        self.v = MLPCritic(obs_dim)

    def step(self, obs):
        '''
        只接受1个obs，用于驱动环境运行
        这个函数是计算出的 old_logpa
        不用梯度，测试的输出该状态下
        使用策略得到的动作， 状态的价值， 动作对应的log p(a)
        '''
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            a = dist.sample()
            logp_a = self.pi._log_prob_from_distribution(dist, a)

            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def act(self, obs):
        '''
        这个函数，仅仅用在ppo_test里面，给一个状态，得到一个动作，用于测试。
        '''
        return self.step(obs)[0]



if __name__ == '__main__':
    import gym
    import gym_2048
    import numpy as np
    from wrapper2048 import ConvFloat
    env = gym.make('2048-v0')
    env = ConvFloat(env)
    ac = MLPActorCritic(env.observation_space,env.action_space)
    obs = env.reset()
    print(ac.step(torch.FloatTensor(obs).unsqueeze(0)))



    
