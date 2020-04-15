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
parser.add_argument('--env', type=str, default='harvest', help='Name of the environment to rollout. Can be '
                                                               'cleanup or harvest.')
parser.add_argument('--algorithm', type=str, default='A3C', help='Name of the rllib algorithm to use.')
parser.add_argument('--num_agents', type=int, default=5, help='Number of agent policies')
parser.add_argument('--num_symbols', type=int, default=9, help='Number of symbols in language')
parser.add_argument('--train_batch_size', type=int, default=26000,
                    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint_frequency', type=int, default=500,
                    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training_iterations', type=int, default=2500, help='Total number of steps to train for')
parser.add_argument('--num_cpus', type=int, default=24, help='Number of available CPUs')
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
          num_agents, num_symbols, grid_search, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    obs_space = None
    act_space = None
    if env == 'harvest':
        obs_space = HarvestEnv.observation_space(num_agents, num_symbols)
        act_space = HarvestEnv.action_space(num_agents, num_symbols)

        def env_creator(env_config):
            return HarvestEnv(env_config)
    else:
        obs_space = CleanupEnv.observation_space(num_agents, num_symbols)
        act_space = CleanupEnv.action_space(num_agents, num_symbols)

        def env_creator(env_config):
            return CleanupEnv(env_config)

    env_name = env + "_env"
    register_env(env_name, env_creator)

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
    config['callbacks']['on_postprocess_traj'] = on_postprocess_traj

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
                "train_batch_size": train_batch_size,
                "sample_batch_size": 50,
                # "batch_mode": "complete_episodes",
                # "metrics_smoothing_episodes": 1,
                "vf_loss_coeff": 0.1,
                "horizon": 1000,
                "gamma": 0.99,
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
                          },
                "env_config": {
                    "num_agents": num_agents,
                    "num_symbols": num_symbols,
                    "obedience_weight": .001,
                    "leadership_weight": .001,
                },
    })

    if args.algorithm == "PPO":
        config.update({"num_sgd_iter": 10,
                       "sgd_minibatch_size": 500,
                       "vf_loss_coeff": 1e-4
        })

    if args.grid_search:
        pass

    return algorithm, env_name, config


def on_postprocess_traj(info):
    # print('TAG on_postprocess_traj')
    # info has keys episode, agent_id, pre_batch, post_batch, all_pre_batches
    # post batch has
    # dict_keys(['t', 'eps_id', 'agent_index', 'obs', 'actions', 'rewards', 'prev_actions',
    # 'prev_rewards', 'dones', 'infos', 'new_obs', 'action_prob', 'action_logp', 'vf_preds',
    # 'state_in_0', 'state_in_1', 'state_out_0', 'state_out_1', 'unroll_id', 'advantages',
    # 'value_targets'])

    # print(info['post_batch']['infos'])
    infos = info['post_batch']['infos']
    intrinsic_vals = [f['intrinsic'] for f in infos]
    env_vals = [f['environmental'] for f in infos]
    # total_vals = [f['intrinsic'] + f['environmental'] for f in infos]

    episode = info['episode']
    # episode.custom_metrics["totals"] = sum(total_vals)
    episode.custom_metrics["envs"] = sum(env_vals)
    episode.custom_metrics["intrinsics"] = sum(intrinsic_vals)


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
                                      args.grid_search,
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
        'checkpoint_at_end': True,
        "config": config,
        'reuse_actors': True,
    }

    analysis = tune.run(**exp_dict)
    print(analysis.get_best_config(metric='custom_metrics/envs_mean'))
