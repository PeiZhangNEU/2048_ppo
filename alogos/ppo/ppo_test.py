# 必须要把根目录的路径添加进来！
import sys
sys.path.append('/home/zp/expand_disk/ShaoWenLU_hmwk/ppo_2048/')  
import gym_2048
import torch
import gym
import time
from wrapper2048 import ConvFloat
from utils_forme import display_video
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_model(path):
    '''把模型加载成cpu形式'''
    model = torch.load(path, map_location=torch.device('cpu'))
    return model

def get_action(model, x, determine=True):
    '''因为model的act，需要传入tensor 的obs，这里写个函数转化'''
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        if determine:
            dist = model.pi._distribution(x)
            logits = dist.logits
            logits = logits.numpy()
            a = np.argmax(logits)
        else:
            a = model.act(x) 
    return a


def test(path, env_name, render=True, num_episodes=500, max_ep_len=10000):
    '''载入模型并且测试'''
    policy = load_model(path)  # 这个载入的policy和logger的save的东西有关
                               # 我save的是ActorCritic这个类，包括类的方法也保留
    env = gym.make(env_name)
    env = ConvFloat(env)
    frams = []
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            img = env.render(mode='rgb_array')
            imgI = Image.fromarray(img)
            # 翻转并且Right90度
            imgI = imgI.transpose(Image.FLIP_LEFT_RIGHT)
            imgI = imgI.rotate(90)
            imgI = np.array(imgI)

            frams.append(imgI)

        a = get_action(policy, o, True)
        o, r, d, info = env.step(a)
        ep_ret += np.exp(16 * r) - 1 
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d\t High %d '%(n, ep_ret, ep_len, info['highest']))
            if info['highest'] > 1024:
                display_video(frams,10, 'ppo_conv_2048_'+str(info['highest'])+'_'+str(ep_ret)+'.gif')
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            frams = []
            n += 1
            


if __name__ == '__main__':
    # test('data/ppo_hopper/ppo_hopper_s0/pyt_save/model.pt','Hopper-v2')
    # test('data/ppo_cartpole/ppo_cartpole_s0/pyt_save/model.pt','CartPole-v0')
    test('ppo_4900_ac.pt','2048-v0')