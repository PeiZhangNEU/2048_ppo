# 把 observation 变形成 256 维度的输入特征。浮点数。
# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import platform
import random
import sys
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import gym
import numpy as np
import torch as th
from gym.error import AlreadyPendingCallError
from gym.vector import SyncVectorEnv
from gym.vector.utils import concatenate, write_to_shared_memory
from gym.wrappers import FlattenObservation, TimeLimit

class FlatFloat(gym.ObservationWrapper):
    '''
    Adds a frame counter to the observation.
    The resulting observation space will be a dictionary, with an additional
    ['time'] entry.
    '''

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                    low=0, high=1, shape=(256,), dtype=np.float32
        )

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.reshape(256,).astype(np.float32)
        return observation

    def step(self, action):
        s_, r, d, info = self.env.step(action)
        s_ = s_.reshape(256,).astype(np.float32)

        # 把奖励归一
        r = np.log(r + 1) /16
        return s_,r,d,info


class ConvFloat(gym.ObservationWrapper):
    '''
    Adds a frame counter to the observation.
    The resulting observation space will be a dictionary, with an additional
    ['time'] entry.
    '''

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                    low=0, high=1, shape=(1,16,16), dtype=np.float32
        )

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.reshape(1, 16, 16).astype(np.float32)
        return observation

    def step(self, action):
        s_, r, d, info = self.env.step(action)
        s_ = s_.reshape(1, 16, 16).astype(np.float32)

        # 把奖励归一
        r = np.log(r + 1) /16
        return s_,r,d,info

