import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

from models.layers.building_blocks import MLP, Conv, DeConv


class PPO_Actor(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        is_discrete: bool,
        activation: nn.Module = nn.Tanh(),
    ):
        super(PPO_Actor, self).__init__()

        self.state_dim = np.prod(input_dim)
        self.action_dim = np.prod(action_dim)

        self.is_discrete = is_discrete

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass for the actor network.
        :param state: Input state tensor.
        :param deterministic: If True, use deterministic action selection.
        :return: Action tensor and additional metadata.
        """
        if self.is_discrete:
            a, metaData = self.discrete_forward(state, deterministic)
        else:
            a, metaData = self.continuous_forward(state, deterministic)

        return a, metaData

    def continuous_forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        if deterministic:
            a = mu
            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)

        else:
            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            a = dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

            entropy = dist.entropy()

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def discrete_forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
            dist = None
            logprobs = torch.zeros_like(logits[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

            entropy = dist.entropy()

        a = F.one_hot(a, num_classes=logits.size(-1))
        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class PPO_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation
        self._dtype = torch.float32

        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
