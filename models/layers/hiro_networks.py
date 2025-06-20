import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from models.base import Base
from models.layers.building_blocks import MLP, Conv, DeConv
from models.layers.ppo_networks import PPO_Actor, PPO_Critic


class TD3_Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        action_min: np.ndarray | None,
        action_max: np.ndarray | None,
        is_discrete: bool,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super(TD3_Actor, self).__init__()

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)
        if action_min is not None and action_max is not None:
            self.action_min = torch.tensor(action_min, device=device)
            self.action_max = torch.tensor(action_max, device=device)
        self.is_discrete = is_discrete

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )

        self.device = device
        self._dummy = torch.tensor(1e-10).to(self.device)
        self.to(self.device)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = True,  # always deterministic
    ):
        if self.is_discrete:
            a, metaData = self.discrete_forward(state, deterministic)
        else:
            a, metaData = self.continuous_forward(state, deterministic)
        return a, metaData

    def continuous_forward(self, state: torch.Tensor, deterministic: bool):
        # Return the deterministic output directly
        mu = self.model(state)

        if deterministic:
            action = mu
        else:
            # Add small exploration noise for action selection (not training!)
            noise = (torch.randn_like(mu) * 0.2).to(self.device)
            action = mu + noise

        if hasattr(self, "action_min") and hasattr(self, "action_max"):
            action = action.clamp(self.action_min, self.action_max).to(torch.float32)

        return action, {
            "dist": self._dummy,
            "probs": self._dummy,
            "logprobs": self._dummy,
            "entropy": self._dummy,
        }

    def discrete_forward(self, state: torch.Tensor, deterministic: bool):
        # Deterministic: just take the argmax
        logits = self.model(state)

        if deterministic:
            # Deterministic TD3-style action selection
            a_idx = torch.argmax(logits, dim=-1)
        else:
            a_idx = self.epsilon_greedy_batch(logits, 0.2, self.device)

        # One-hot action vector
        a = F.one_hot(a_idx, num_classes=logits.size(-1)).float()

        return a, {
            "dist": self._dummy,
            "probs": self._dummy,
            "logprobs": self._dummy,
            "entropy": self._dummy,
        }

    def epsilon_greedy_batch(self, logits, epsilon, device):
        batch_size, num_actions = logits.shape

        # Exploitation: greedy actions
        greedy_actions = torch.argmax(logits, dim=-1)

        # Exploration: random actions
        random_actions = torch.randint(0, num_actions, (batch_size,), device=device)

        # Mask: for each sample, decide if we explore (1) or exploit (0)
        explore_mask = torch.rand(batch_size, device=device) < epsilon

        # Final action: choose random where explore_mask is True, otherwise greedy
        a_idx = torch.where(explore_mask, random_actions, greedy_actions)

        return a_idx

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        raise NotImplementedError("Deterministic TD3 actor does not use log_prob")

    def entropy(self, dist: torch.distributions):
        raise NotImplementedError("Deterministic TD3 actor does not use entropy")


class HIRO_Policy(Base):
    def __init__(
        self,
        state_dim: tuple | int,
        goal_dim: tuple | int,
        action_dim: tuple | int,
        actor_fc_dim: list,
        critic_fc_dim: list,
        is_discrete: bool,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        expl_noise: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        gamma: float = 0.99,
        policy_freq: int = 2,
        tau: float = 0.005,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.actor_input_dim = np.prod(state_dim) + np.prod(goal_dim)
        self.critic_input_dim = (
            np.prod(state_dim) + np.prod(goal_dim) + np.prod(action_dim)
        )
        self.action_dim = int(np.prod(action_dim))

        self.actor = TD3_Actor(
            input_dim=self.actor_input_dim,
            hidden_dim=actor_fc_dim,
            action_dim=self.action_dim,
            action_min=action_min,
            action_max=action_max,
            activation=activation,
            is_discrete=is_discrete,
            device=device,
        )
        self.actor_target = TD3_Actor(
            input_dim=self.actor_input_dim,
            hidden_dim=actor_fc_dim,
            action_dim=self.action_dim,
            action_min=action_min,
            action_max=action_max,
            activation=activation,
            is_discrete=is_discrete,
            device=device,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = PPO_Critic(
            input_dim=self.critic_input_dim,
            hidden_dim=critic_fc_dim,
            activation=activation,
        )
        self.critic2 = PPO_Critic(
            input_dim=self.critic_input_dim,
            hidden_dim=critic_fc_dim,
            activation=activation,
        )

        self.critic1_target = PPO_Critic(
            input_dim=self.critic_input_dim,
            hidden_dim=critic_fc_dim,
            activation=activation,
        )
        self.critic2_target = PPO_Critic(
            input_dim=self.critic_input_dim,
            hidden_dim=critic_fc_dim,
            activation=activation,
        )

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=critic_lr
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=critic_lr
        )

        self._initialize_target_networks()

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.device = device
        self.total_it = 0
        self.to(self.device).to(self.dtype)

    def _initialize_target_networks(self):
        """
        Match the weight and bias of non-target network to the targets for consistency
        """
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)

    def _update_target_network(self, target: nn.Module, origin: nn.Module, tau: float):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data
            )

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = True,
    ):
        """
        Forward pass for the actor network.
        :param state: Input state tensor.
        :param deterministic: If True, use deterministic action selection.
        :return: Action tensor and additional metadata.
        """
        state = self.preprocess_state(state)
        goal = self.preprocess_state(goal)

        actor_state = torch.cat([state, goal], dim=-1)
        a, metaData = self.actor(actor_state, deterministic)
        return a, metaData

    def learn(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_goals: torch.Tensor,
        terminals: torch.Tensor,
        name: str,
    ):
        # prepare ingredients
        states = self.preprocess_state(states)
        next_states = self.preprocess_state(next_states)
        goals = self.preprocess_state(goals)
        next_goals = self.preprocess_state(next_goals)

        with torch.no_grad():
            actor_next_states = torch.cat([next_states, next_goals], dim=-1)

            next_actions, _ = self.actor_target(actor_next_states, deterministic=False)
            next_actions = next_actions.to(self.dtype)

            critic_next_states = torch.cat(
                [next_states, next_goals, next_actions], dim=-1
            )

            target_Q1 = self.critic1_target(critic_next_states)
            target_Q2 = self.critic2_target(critic_next_states)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = (rewards + (1 - terminals) * self.gamma * target_Q).detach()

        critic_states = torch.cat([states, goals, actions], dim=-1)

        current_Q1 = self.critic1(critic_states)
        current_Q2 = self.critic2(critic_states)

        critic1_loss = F.smooth_l1_loss(current_Q1, target_Q)
        critic2_loss = F.smooth_l1_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        td_error = (target_Q - current_Q1).mean().cpu()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        critic_grad_dict = self.compute_gradient_norm(
            [self.critic1, self.critic2],
            ["critic1", "critic2"],
            dir=f"{name}",
            device=self.device,
        )
        critic_norm_dict = self.compute_weight_norm(
            [self.critic1, self.critic2, self.critic1_target, self.critic2_target],
            ["critic1", "critic2", "critic1_target", "critic2_target"],
            dir=f"{name}",
            device=self.device,
        )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if (self.total_it + 1) % self.policy_freq == 0:
            actor_states = torch.cat([states, goals], dim=-1)

            a, _ = self.actor(actor_states, deterministic=True)
            critic_states = torch.cat([states, goals, a], dim=-1)
            # with torch.no_grad():
            Q1 = self.critic1(critic_states)
            actor_loss = -Q1.mean()  # Deterministic TD3-style

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            actor_grad_dict = self.compute_gradient_norm(
                [self.actor],
                ["actor"],
                dir=f"{name}",
                device=self.device,
            )
            actor_norm_dict = self.compute_weight_norm(
                [self.actor, self.actor_target],
                ["actor", "actor_target"],
                dir=f"{name}",
                device=self.device,
            )
            self.actor_optimizer.step()

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            loss_dict = {
                f"{name}/actor_loss": actor_loss.item(),
                f"{name}/critic_loss": critic_loss.item(),
            }
            loss_dict.update(actor_grad_dict)
            loss_dict.update(actor_norm_dict)
        else:
            loss_dict = {f"{name}/critic_loss": critic_loss.item()}

        loss_dict[f"{name}/avg_batch_rewards"] = rewards.mean().item()
        loss_dict[f"{name}/td_error"] = td_error.item()

        loss_dict.update(critic_grad_dict)
        loss_dict.update(critic_norm_dict)

        self.total_it += 1

        return loss_dict


class HL_Policy(HIRO_Policy):
    def __init__(self, scale_high, **kwargs):
        super().__init__(**kwargs)

        self.scale = scale_high
        self.name = "HL_Policy"

    def off_policy_corrections(
        self,
        low_con: nn.Module,
        batch_size: int,
        sgoals: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        candidate_goals: int = 8,
    ):
        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) - np.array(first_s))[
            :, np.newaxis, : self.action_dim
        ]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals

        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(
            loc=diff_goal,
            scale=0.5 * self.scale[None, None, :],
            size=(batch_size, candidate_goals, original_goal.shape[-1]),
        )
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:, c]
            candidate = (subgoal + states[:, 0, : self.action_dim])[:, None] - states[
                :, :, : self.action_dim
            ]
            candidate = candidate.reshape(*goal_shape)
            with torch.no_grad():
                action, _ = low_con(observations, candidate, deterministic=True)
            policy_actions[c] = action.cpu().numpy()

        difference = policy_actions - true_actions
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape(
            (ncands, batch_size, seq_len) + action_dim
        ).transpose(1, 0, 2, 3)

        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return (
            candidates[np.arange(batch_size), max_indices],
            logprob[max_indices, :].mean(),
        )

    def learn(self, ll_policy: nn.Module, replay_buffer):
        (
            states,
            goals,
            actions,
            next_states,
            rewards,
            terminals,
            states_arr,
            actions_arr,
        ) = replay_buffer.sample()

        states = states.reshape(states.shape[0], -1)
        actions = actions.reshape(actions.shape[0], -1)
        next_states = next_states.reshape(next_states.shape[0], -1)
        states_arr = states_arr.reshape(states_arr.shape[0], states_arr.shape[1], -1)
        goals = goals.reshape(goals.shape[0], -1)

        actions, logprobs = self.off_policy_corrections(
            ll_policy,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy(),
        )

        actions = torch.from_numpy(actions).to(self.device).to(self.dtype)
        loss_dict = super().learn(
            states, goals, actions, rewards, next_states, goals, terminals, self.name
        )
        loss_dict[f"{self.name}/off-policy_correction_logprobs"] = logprobs

        return loss_dict


class LL_policy(HIRO_Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "LL_Policy"

    def learn(self, replay_buffer):
        states, goals, actions, next_states, next_goals, rewards, terminals = (
            replay_buffer.sample()
        )
        return super().learn(
            states,
            goals,
            actions,
            rewards,
            next_states,
            next_goals,
            terminals,
            self.name,
        )
