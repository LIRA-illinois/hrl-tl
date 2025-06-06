import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from utils.sampler import Sampler


class HCTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
        sampler: Sampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        seed: int = 0,
    ) -> None:

        # environment to sample and run
        self.env = env
        # the policy that wraps the actor and critic with their correspoding learning module
        self.policy = policy
        # sample receives the env and policy to collect samples stochastic manner
        self.sampler = sampler

        # logger is wandb logger
        self.logger = logger
        # writer is tensorboard writer
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        # How many episodes will be used to evaluate the performance?
        self.eval_num = eval_num
        # Given timesteps, how many times we want to evaluate?
        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_min_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                loss_dict, timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / step
                remaining_time = avg_time_per_iter * (self.timesteps - step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step + timesteps
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if step >= self.eval_interval * eval_idx:
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, running_video = self.evaluate()

                    # Manual logging
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

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, infos = self.env.reset(seed=self.seed)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    [_, a], _ = self.policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0:
                    # Plotting
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(a)
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
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
        name = f"hl_model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        hl_model = self.policy.hl_actor

        if hl_model is not None:
            hl_model = deepcopy(hl_model).to("cpu")
            torch.save(hl_model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_hl_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(hl_model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

        name = f"ll_model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        ll_model = self.policy.ll_actor

        if ll_model is not None:
            ll_model = deepcopy(ll_model).to("cpu")
            torch.save(ll_model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_ll_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(ll_model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
