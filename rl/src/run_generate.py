from pearl.utils.functional_utils.train_and_eval.online_learning import run_episode
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent
from pearl.user_envs.wrappers.gym_avg_torque_cost import GymAvgTorqueWrapper
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
import gymnasium as gym
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.neural_networks.sequential_decision_making.actor_networks import VanillaContinuousActorNetwork
from pearl.neural_networks.sequential_decision_making.q_value_networks import VanillaQValueNetwork
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (
    NormalDistributionExploration,
)
from pearl.safety_modules.reward_constrained_safety_module import (
    RCSafetyModuleCostCriticContinuousAction,
)

from matplotlib import pyplot as plt
import torch
import numpy as np
import pickle
from env_boxaction import MaskedGoEnv
from data_io.file_io import data_to_jsonl_append

corpus_path = "/path/to/corpus.jsonl"
checkpoint_path = "/path/to/checkpoint"
output_path = "/path/to/output.jsonl"
number_of_generated_programs = 5


set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = MaskedGoEnv(corpus_path)

with open(checkpoint_path, 'rb') as f:
    loaded_rctd3_agent = pickle.load(f)


number_of_episodes = number_of_generated_programs
print_every_x_episodes = 1

total_steps = 0
total_episodes = 0
while True:
    if total_episodes >= number_of_episodes:
        break
    old_total_steps = total_steps
    episode_info, episode_total_steps = run_episode(
        loaded_rctd3_agent,
        env,
        learn=False,
        exploit=False,
    )
    data_to_jsonl_append(output_path, {"code":env.state})
    total_steps += episode_total_steps
    total_episodes += 1
    if total_episodes % print_every_x_episodes == 0:
        print(
                f"episode {total_episodes}, agent={loaded_rctd3_agent}, env={env}",
            )
    