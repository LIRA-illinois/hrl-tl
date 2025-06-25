from io import BytesIO

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image


class HIROWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(HIROWrapper, self).__init__(env)

    def reset(self, **kwargs):
        state_dict, info = self.env.reset(**kwargs)
        # state_dict = {"observation": state, "desired_goal": self.desired_goal}

        return state_dict, info

    def step(self, action):
        # Call the original step method
        state_dict, reward, termination, truncation, info = self.env.step(action)
        # state_dict = {"observation": state, "desired_goal": self.desired_goal}

        done = termination or truncation

        return state_dict, reward, done, info

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)
