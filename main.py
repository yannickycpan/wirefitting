"""
How to use:
sample file: MC_SARSA.json
[begin MC_SARSA.json]
{
    "environment": "MountainCar",
    "agent": "SARSA",
    "representation": "tile_code",
    "episodes": 300,
    "representation_parameters": {
        "tilings": 8,
        "tiles": 8,
        "mem_size": 2048,
        "pairs": 0
    },
    "sweeps": {
        "alpha": [0.1, 0.2, 0.3],
        "gamma": [1.0],
        "lambda": [0.95],
        "epsilon": [0, 0.1, 0.2]
    }
}
[end MC_SARSA.json]

call: python run.py MC_SARSA.json 0
the 0th parameter setting (in this case alpha=0.1 and epsilon = 0) will be run
continue calling and incrementing the index until all parameters have been run.
To get multiple runs of the same parameter settings roll over the index.
For instance, in this example there are 9 possible parameter settings: 3 * 1 * 1 * 3 = 9
So if you run: python run.py MC_SARSA.json 9 (remember 0 based indexing)
this will again run the parameter setting (alpha=0.1 and epsilon = 0)
"""
from utils.rl_exp import Experiment
from utils.AtariExperiment import Atari
import numpy as np
import os
import sys
import json

save_file_format = '%10.6f'


def get_sweep_parameters(parameters, index):
    out = {}
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out[key] = parameters[key][int((index / accum) % num)]
        accum *= num
    n_run = int(index / accum)
    n_setting = int(index % accum)
    return out, n_run, n_setting


def get_count_parameters(parameters):
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        accum *= num
    return accum


def merge_agent_params(agent_params, sweep_params):
    for key in sweep_params:
        agent_params[key] = sweep_params[key]
    return agent_params


def supplement_common_params(agent_params, env_params, all_params):
    for key in ["warmUpSteps", "bufferSize", "batchSize", "evalEverySteps",
                "numTrainEpisodes","numEvalEpisodes","maxTotalSamples"]:
        if key not in agent_params:
            agent_params[key] = all_params[key]
    agent_params['type'] = env_params['type'] if 'type' in env_params else None


# can add if condition in this function if some experiment has special setting
def make_experiment(agent_params, env_params, seed):
    if 'type' in env_params and env_params['type'] == 'Atari':
        return Atari(agent_params, env_params, seed)
    return Experiment(agent_params, env_params, seed)


def save_results(env_name, agent_name, sweep_params, train_lc, eval_lc, eval_ste, n_setting, n_run):
    storedir = env_name + 'results/'
    prefix = storedir + env_name + '_' + agent_name + '_setting_' + str(n_setting) + '_run_' + str(n_run)

    name = prefix + '_EpisodeLC.txt'
    train_lc.tofile(name, sep=',', format=save_file_format)

    name = prefix + '_EpisodeEvalLC.txt'
    eval_lc.tofile(name, sep=',', format=save_file_format)

    name = prefix + '_EpisodeEvalSte.txt'
    eval_ste.tofile(name, sep=',', format=save_file_format)

    params = []
    params_names = '_'
    for key in sweep_params:
        params.append(sweep_params[key])
        params_names += (key + '_')
    params = np.array(params)
    name = prefix + params_names + 'Params.txt'
    params.tofile(name, sep=',', format=save_file_format)
    return None


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('run as "python run.py [json_file] [index]')
        exit(0)
    # load experiment description from json file
    file = sys.argv[1]
    index = int(sys.argv[2])
    json_dat = open(file, 'r')
    exp = json.load(json_dat)
    json_dat.close()
    '''
    #file = 'dqnoption.json'
    #file = 'optimal.json'
    #file = 'ddpg.json'
    file = 'jsonfiles/dqn.json'
    json_dat = open(file, 'r')
    exp = json.load(json_dat)
    json_dat.close()
    index = 0
    '''
    accm_total_runs = 0
    last_accm_total_runs = 0
    avg_runs = 1
    for env_name in exp['environment_names']:
        for agent_name in exp['agent_names']:
            dirname = env_name + 'results'
            if os.path.exists(dirname) == False:
                os.makedirs(dirname)
            # environment parameter
            env_params = exp[env_name]
            env_params['name'] = env_name
            # generate agent parameter
            agent_params = exp[agent_name]
            supplement_common_params(agent_params, env_params, exp)

            n_total_runs = get_count_parameters(agent_params['sweeps'])*avg_runs
            accm_total_runs += n_total_runs
            if index >= accm_total_runs:
                #print('acc total runs is ', n_total_runs, accm_total_runs)
                last_accm_total_runs = accm_total_runs
                continue
            else:
                index -= last_accm_total_runs

            agent_sweep_params, n_run, n_setting = get_sweep_parameters(agent_params['sweeps'], index)
            print('-----------the final corrected index is-------------------', index)
            print('------------setting and run indexes are-----------------', env_name, agent_name, n_setting, n_run)

            agent_params['name'] = agent_name
            agent_params = merge_agent_params(agent_params, agent_sweep_params)
            # create experiment
            experiment = make_experiment(agent_params, env_params, seed = n_run)
            # run experiment and save result
            train_rewards, eval_rewards, eval_ste = experiment.run()
            save_results(env_name, agent_name, agent_sweep_params, train_rewards, eval_rewards, eval_ste, n_setting, n_run)
            exit(0)
