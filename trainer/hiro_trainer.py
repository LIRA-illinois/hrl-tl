import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from utils.sampler import Sampler


class HiroTrainer:
    def __init__(
        self,
        env,
        policy,
        logger,
        writer,
        epochs: int = 20000,
        eval_num: int = 10,
        seed: int = 0,
    ):
        self.env = env
        self.policy = policy
        self.logger = logger
        self.writer = writer

        self.epochs = epochs
        self.eval_interval = int(self.epochs // 100)
        self.eval_num = eval_num

        self.last_min_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.seed = seed

    def train(self):
        global_step = 0
        eval_idx = 0

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        with tqdm(
            total=self.epochs,
            desc=f"{self.policy.name} Training (Epochs)",
        ) as pbar:
            while pbar.n < self.epochs:
                e = pbar.n + 1  # + 1 to avoid zero division

                obs, _ = self.env.reset()

                fg = obs["desired_goal"]
                s = obs["observation"]

                done = False

                step = 0
                episode_reward = 0

                self.policy.set_final_goal(fg)

                loss_dict_list = []
                while not done:
                    # Take action
                    a, r, n_s, done = self.policy.step(
                        s, self.env, step, global_step, explore=True
                    )

                    # Append
                    self.policy.append(step, s, a, n_s, r, done)

                    # Train
                    loss_dict = self.policy.learn(global_step)
                    loss_dict_list.append(loss_dict)

                    # Updates
                    s = n_s
                    episode_reward += r
                    step += 1
                    global_step += 1
                    self.policy.end_step()

                self.policy.end_episode()
                pbar.update(1)

                self.write_log(
                    self.average_dict_values(loss_dict_list), step=global_step
                )

                #### EVALUATIONS ####
                if step >= self.eval_interval * eval_idx:
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.evaluate()

                    # Manual logging
                    # fig, ax = plt.subplots()
                    # ax.stem()

                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_video(
                        running_video,
                        step=step,
                        logdir=f"videos",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(step)

                torch.cuda.empty_cache()

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            step = 0
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            fg = state["desired_goal"]
            state = state["observation"]

            self.policy.set_final_goal(fg)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    a, rew, next_state, done = self.policy.step(state, self.env, step)

                if num_episodes == 0:
                    # Plotting
                    image = self.env.render()
                    image_array.append(image)

                state = next_state
                ep_reward.append(rew)

                self.policy.end_step()

                if done:
                    self.policy.end_episode()
                    ep_buffer.append(
                        {
                            "return": self.discounted_returns(
                                ep_reward, self.policy.gamma
                            ),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
        }

        return eval_dict, image_array

    def discounted_returns(self, rewards, gamma):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        tensor = np.stack(image, axis=0)
        video_path = os.path.join(logdir, name)
        self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict
