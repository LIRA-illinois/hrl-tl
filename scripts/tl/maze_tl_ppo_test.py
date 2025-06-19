import os
from typing import Any

import gym_multigrid
import gymnasium as gym
import imageio
from gym_tl_tools import Predicate, TLObservationReward, replace_special_characters
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from hrl_tl.envs.tl_fourroom import var_value_info_generator

if __name__ == "__main__":
    spec_id: int = 2

    tl_specs: list[str] = [
        "F(psi_ld) & G(!psi_lv)",
        "F(psi_td) & G(!psi_hl)",
        "F(psi_gl)",
    ]
    tl_spec: str = tl_specs[spec_id]
    predicates: list[Predicate] = [
        Predicate(name="psi_ld", formula="d_ld < 0.5"),
        Predicate(name="psi_bd", formula="d_bd < 0.5"),
        Predicate(name="psi_td", formula="d_td < 0.5"),
        Predicate(name="psi_rd", formula="d_rd < 0.5"),
        Predicate(name="psi_gl", formula="d_gl < 0.5"),
        Predicate(name="psi_lv", formula="d_lv < 0.5"),
        Predicate(name="psi_hl", formula="d_hl < 0.5"),
    ]
    retrain_model: bool = True
    gpu_id: int = 1
    model_name: str = "maze_tl_ppo_test_rand_" + replace_special_characters(tl_spec)
    model_save_dir: str = "out/poc/low-level"
    total_timesteps: int = 50_000
    max_episode_steps: int = 100
    n_envs: int = 10
    callback_save_frequency: int = int(total_timesteps / 10 / n_envs)
    batch_size: int = 1_000
    rl_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "learning_rate": 0.0003,
        "n_steps": int(batch_size / n_envs),
        "batch_size": batch_size,
        "n_epochs": 40,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "stats_window_size": 100,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    env_kwargs = {
        "max_episode_steps": max_episode_steps,
        "spawn_type": spec_id,
        "random_init_pos": True,
        "layout_config": {
            "field_map": [
                "#############",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#     #     #",
                "## ####     #",
                "#     ### ###",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#############",
            ],
            "spawn_configs": [
                {
                    "agent": (3, 9),
                    "goal": {"pos": (9, 4), "reward": 1.0},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0},
                        {"pos": (9, 2), "reward": 0},
                        {"pos": (11, 1), "reward": 0},
                        {"pos": (5, 3), "reward": 0},
                        {"pos": (3, 5), "reward": 0},
                        {"pos": (3, 2), "reward": 0},
                        {"pos": (5, 9), "reward": -1},
                        {"pos": (3, 8), "reward": -1},
                        {"pos": (2, 11), "reward": -1},
                        {"pos": (10, 8), "reward": -1},
                        {"pos": (8, 9), "reward": -1},
                        {"pos": (7, 11), "reward": -1},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0},
                        {"pos": (10, 5), "reward": 0},
                        {"pos": (8, 6), "reward": 0},
                        {"pos": (4, 4), "reward": -1},
                        {"pos": (2, 3), "reward": -1},
                        {"pos": (1, 1), "reward": -1},
                        {"pos": (2, 7), "reward": 0},
                        {"pos": (1, 9), "reward": 0},
                        {"pos": (4, 10), "reward": 0},
                        {"pos": (7, 8), "reward": -1},
                        {"pos": (9, 10), "reward": -1},
                        {"pos": (11, 11), "reward": -1},
                    ],
                },
                {
                    "agent": (2, 6),
                    "goal": {"pos": (9, 4), "reward": 1.0},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0},
                        {"pos": (9, 2), "reward": 0},
                        {"pos": (11, 1), "reward": 0},
                        {"pos": (5, 3), "reward": 0},
                        {"pos": (3, 5), "reward": 0},
                        {"pos": (3, 2), "reward": 0},
                        {"pos": (5, 9), "reward": -1},
                        {"pos": (3, 8), "reward": -1},
                        {"pos": (2, 11), "reward": -1},
                        {"pos": (10, 8), "reward": -1},
                        {"pos": (8, 9), "reward": -1},
                        {"pos": (7, 11), "reward": -1},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0},
                        {"pos": (10, 5), "reward": 0},
                        {"pos": (8, 6), "reward": 0},
                        {"pos": (4, 4), "reward": -1},
                        {"pos": (2, 3), "reward": -1},
                        {"pos": (1, 1), "reward": -1},
                        {"pos": (2, 7), "reward": 0},
                        {"pos": (1, 9), "reward": 0},
                        {"pos": (4, 10), "reward": 0},
                        {"pos": (7, 8), "reward": -1},
                        {"pos": (9, 10), "reward": -1},
                        {"pos": (11, 11), "reward": -1},
                    ],
                },
                {
                    "agent": (6, 3),
                    "goal": {"pos": (9, 4), "reward": 1.0},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0},
                        {"pos": (9, 2), "reward": 0},
                        {"pos": (11, 1), "reward": 0},
                        {"pos": (5, 3), "reward": 0},
                        {"pos": (3, 5), "reward": 0},
                        {"pos": (3, 2), "reward": 0},
                        {"pos": (5, 9), "reward": -1},
                        {"pos": (3, 8), "reward": -1},
                        {"pos": (2, 11), "reward": -1},
                        {"pos": (10, 8), "reward": -1},
                        {"pos": (8, 9), "reward": -1},
                        {"pos": (7, 11), "reward": -1},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0},
                        {"pos": (10, 5), "reward": 0},
                        {"pos": (8, 6), "reward": 0},
                        {"pos": (4, 4), "reward": -1},
                        {"pos": (2, 3), "reward": -1},
                        {"pos": (1, 1), "reward": -1},
                        {"pos": (2, 7), "reward": 0},
                        {"pos": (1, 9), "reward": 0},
                        {"pos": (4, 10), "reward": 0},
                        {"pos": (7, 8), "reward": -1},
                        {"pos": (9, 10), "reward": -1},
                        {"pos": (11, 11), "reward": -1},
                    ],
                },
            ],
        },
    }
    wrapper_kwargs = {
        "tl_spec": tl_spec,
        "atomic_predicates": predicates,
        "var_value_info_generator": var_value_info_generator,
        "reward_config": {
            "terminal_state_reward": 0,
            "state_trans_reward_scale": 10,
            "dense_reward": True,
            "dense_reward_scale": 0.01,
        },
        "early_termination": True,
    }

    model_save_path: str = os.path.join(model_save_dir, model_name)
    animation_save_dir: str = os.path.join(model_save_path)

    # env = gym.make(
    #     "multigrid-rooms-v0", max_episode_steps=max_episode_steps, spawn_type=3
    # )
    # env = TLObservationReward(env, **wrapper_kwargs)

    env = make_vec_env(
        "multigrid-rooms-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
        wrapper_class=TLObservationReward,
        wrapper_kwargs=wrapper_kwargs,
    )

    env_kwargs["random_init_pos"] = False  # Disable random init pos for demo env
    demo_env = gym.make(
        "multigrid-rooms-v0",
        render_mode="rgb_array",
        **env_kwargs,
    )
    demo_env = TLObservationReward(
        demo_env,
        **wrapper_kwargs,
    )
    if not os.path.exists(model_save_path) or retrain_model:
        os.makedirs(model_save_dir, exist_ok=True)
        model = PPO(
            **rl_config,
            env=env,
            verbose=1,
            tensorboard_log=os.path.join(model_save_path, "tb"),
            device="cuda:{}".format(gpu_id),
        )
        model.learn(total_timesteps=total_timesteps)
        env.close()
        # Save the model
        model.save(os.path.join(model_save_path, "final_model"))
    else:
        print(f"Model {model_name} already exists, loading...")
        model = PPO.load(os.path.join(model_save_path, "final_model"), env=demo_env)
    # Save the animation

    # Video generation using imageio
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        # Ensure action is a numpy int64 scalar
        obs, reward, terminated, truncated, info = demo_env.step(action)
        print(
            f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info['is_success']}"
        )
        rewards.append(reward)
        frame = demo_env.render()
        frames.append(frame)
    demo_env.close()
    print(f"Total reward: {sum(rewards)}")
    video_path = os.path.join(animation_save_dir, f"{model_name}.gif")
    imageio.mimsave(video_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
