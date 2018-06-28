import numpy as np
import random
from environments.environments import Environment, BlockWorld, AtariEnvironment
from agents.WireFitting import WireFittingAgent

def make_environment(env_params):
    env_name = env_params['name']
    if env_name == 'BlockWorld':
        return BlockWorld(env_params)
    elif 'type' in env_params and env_params['type'] == 'Atari':
        return AtariEnvironment(env_params)
    else:
        return Environment(env_params)


def make_agent(agent_params):
    agent_name = agent_params['name']
    if agent_name == "WireFitting":
        return WireFittingAgent(agent_params)
    else:
        print("agent not found!!!")
        exit(0)


def supplement_agent_params(agent_params, env):
    agent_params['stateDim'] = env.stateDim
    agent_params['actionDim'] = env.actionDim
    agent_params['actionBound'] = env.actionBound
    #print('------------------agent params are -----------------------', agent_params)
    return agent_params


class Experiment(object):
    def __init__(self, agent_params, env_params, seed):
        agent_params['seed'] = seed
        env_params['seed'] = seed
        self.environment = make_environment(env_params)

        agent_params = supplement_agent_params(agent_params, self.environment)
        self.agent = make_agent(agent_params)

        self.MaxEpisodeSteps = self.environment.EPISODE_STEPS_LIMIT
        self.NumEpisodes = agent_params['numTrainEpisodes']
        self.NumEvalEpisodes = agent_params['numEvalEpisodes']
        self.EvalEverySteps = agent_params['evalEverySteps']
        self.NumTotalSamples = agent_params['maxTotalSamples']

        if self.NumEvalEpisodes > 0:
            self.eval_environment = make_environment(env_params)
        else:
            self.eval_environment = None

        np.random.seed(seed)
        random.seed(seed)

        self.train_step_rewards = []
        self.eval_step_rewards = []
        self.train_episode_rewards = []
        self.eval_episode_rewards = []
        self.eval_episode_rewards_ste = []

        self.accum_reward = 0.
        self.episode = 0

        self.sampleCount = 0

    # Runs a single episode
    def TrainEpisode(self):
        episode_reward = 0.
        step = 0
        done = False
        obs = self.environment.reset()
        act = self.agent.take_action(obs)
        while not (done or step == self.MaxEpisodeSteps):
            if self.sampleCount % self.EvalEverySteps == 0:
                self.EvalRun()
            obs_n, reward, done, _ = self.environment.step(act)
            self.accum_reward += reward
            episode_reward += reward
            self.agent.update(obs, act, obs_n, float(reward), done)
            act = self.agent.take_action(obs_n)
            obs = obs_n
            step += 1
            self.sampleCount += 1
        return episode_reward, step

    def EvalEpisode(self):
        episode_reward = 0.
        step = 0
        done = False
        obs = self.eval_environment.reset()
        act = self.agent.take_action_eval(obs)
        while not (done or step == self.MaxEpisodeSteps):
            obs_n, reward, done, info = self.eval_environment.step(act)
            episode_reward += reward
            act = self.agent.take_action_eval(obs_n)
            step += 1
        return episode_reward

    def EvalRun(self):
        '''start evaluation episodes'''
        eval_rewards = np.zeros(self.NumEvalEpisodes)
        for epi in range(self.NumEvalEpisodes):
            eval_rewards[epi] = self.EvalEpisode()
        self.eval_episode_rewards.append(np.mean(eval_rewards))
        self.eval_episode_rewards_ste.append(np.std(eval_rewards) / np.sqrt(self.NumEvalEpisodes))

    def run(self):
        every = 1
        for episode in range(self.NumEpisodes):
            # runs a single episode and returns the accumulated reward for that episode
            reward, num_steps = self.TrainEpisode()
            self.train_episode_rewards.append(reward)
            if episode % every == 0:
                print("ep: "+ str(episode) + ", r: " + str(reward), "num steps: ", str(num_steps))
                if len(self.eval_episode_rewards) > 1:
                    print("eval r: " + str(self.eval_episode_rewards[-1]))
            if self.sampleCount >= self.NumTotalSamples:
                break
        self.environment.close()
        self.eval_environment.close()
        #self.agent.save_model(self.environment.name, self.agent.name)
        return np.array(self.train_episode_rewards), \
               np.array(self.eval_episode_rewards),\
               np.array(self.eval_episode_rewards_ste)
