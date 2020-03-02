import argparse

import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog

from ray.tune import run_experiments
from ray.tune.registry import register_env

import tensorflow as tf

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.obedience_model import ObedienceLSTM


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help='Name experiment will be stored under')
parser.add_argument('--env', type=str, default='cleanup', help='Name of the environment to rollout. Can be '
                                                               'cleanup or harvest.')
parser.add_argument('--algorithm', type=str, default='A3C', help='Name of the rllib algorithm to use.')
parser.add_argument('--num_agents', type=int, default=5, help='Number of agent policies')
parser.add_argument('--num_symbols', type=int, default=3, help='Number of symbols in language')
parser.add_argument('--train_batch_size', type=int, default=26000,
                    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint_frequency', type=int, default=50,
                    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training_iterations', type=int, default=50, help='Total number of steps to train for')
parser.add_argument('--num_cpus', type=int, default=2, help='Number of available CPUs')
parser.add_argument('--num_gpus', type=int, default=0, help='Number of available GPUs')
parser.add_argument('--use_gpus_for_workers', action='store_true', default=False,
                    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument('--use_gpu_for_driver', action='store_true', default=False,
                    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument('--num_workers_per_device', type=float, default=2,
                    help='Number of workers to place on a single device (CPU or GPU)')
parser.add_argument('--num_envs_per_worker', type=float, default=1,
                    help='Number of envs to place on a single worker')
#parser.add_argument('--multi_node', action='store_true', default=False,
#                    help='If true the experiments are run in multi-cluster mode')
parser.add_argument('--local_mode', action='store_true', default=False,
                    help='Force all the computation onto the driver. Useful for debugging.')
#parser.add_argument('--eager_mode', action='store_true', default=False,
#                    help='Perform eager execution. Useful for debugging.')
parser.add_argument('--use_s3', action='store_true', default=False,
                    help='If true upload to s3')
parser.add_argument('--grid_search', action='store_true', default=False,
                    help='If true run a grid search over relevant hyperparams')

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176}

MODEL_NAME = "conv_to_fc_net"


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, num_symbols, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents, num_symbols=num_symbols)
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents, num_symbols=num_symbols)

    env_name = env + "_env"
    register_env(env_name, env_creator)

    env_instance = env_creator('')
    obs_space = env_instance.observation_space
    act_space = env_instance.action_space

    # register the custom model
    ModelCatalog.register_custom_model(MODEL_NAME, ObedienceLSTM)

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {'custom_model': MODEL_NAME}

    # Setup with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # gets the A3C trainer and its default config
    # source at https://github.com/ray-project/ray/blob/d537e9f0d8b84414a2aba7a7d0a68d59241f1490/rllib/agents/a3c/a3c.py
    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = env_creator
    config['env_config']['env_name'] = env_name
    # config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                #"train_batch_size": train_batch_size,
                "sample_batch_size": 50,
                "vf_loss_coeff": 0.1,
                "horizon": 1000,
                # "gamma": 0.99,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": num_gpus,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": policy_mapping_fn,
                },
                "model": {"custom_model": MODEL_NAME,
                          #"custom_preprocessor": "nothing",
                          "use_lstm": False,
                          "custom_options": {
                              "num_agents": num_agents,
                              "num_symbols": num_symbols,
                              "fcnet_hiddens": [32, 32],
                              "cell_size": 128,
                          },
                          "conv_filters": [[6, [3, 3], 1]],
                          #"lstm_cell_size": 128
                          # conv filters??
                          }
                # bunch of other custom stuff wrapped in tune
    })

    if args.grid_search:
        # TODO: even more tune stuff
        pass

    return algorithm, env_name, config


# def main(unused_argv):
if __name__ == '__main__':
    args = parser.parse_args()
    if args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()

    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params

    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents, args.num_symbols,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    exp_name = f'{args.env}_{args.algorithm}' if not args.exp_name else args.exp_name

    config['env'] = env_name

    # custom trainer is called into here

    exp_dict = {
        'name': exp_name,
        # 'run_or_experiment': trainer,
        'run_or_experiment': args.algorithm,
        "stop": {
            "training_iteration": args.training_iterations
        },
        'checkpoint_freq': args.checkpoint_frequency,
        "config": config,
    }

    tune.run(**exp_dict)
