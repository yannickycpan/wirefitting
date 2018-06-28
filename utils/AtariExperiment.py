from utils.rl_exp import Experiment
import numpy as np

class Atari(Experiment):
    def __init__(self, agent_params, env_params, seed):
        super(Atari, self).__init__(agent_params, env_params, seed)

    # Runs a single episode
    def TrainEpisode(self):
        episode_reward = 0.
        step = 0
        done = False
        obs = self.environment.reset()
        act = self.agent.take_action(obs)
        while not (done or step == self.MaxEpisodeSteps or self.sampleCount == self.NumTotalSamples):
            obs_n, reward, done, _ = self.environment.step(act)
            #print('------------one step reward is ----------------', reward)
            #print(' ---------state first three frames sum -----------', np.sum(obs[:,:,1:]), obs.max())
            #print(' ---------state last three frames sum -----------', np.sum(obs_n[:,:,:3]), obs.min())
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
        obs = self.environment.reset()
        act = self.agent.take_action(obs)
        while not (done or step == self.MaxEpisodeSteps):
            obs_n, reward, done, info = self.environment.step(act)
            episode_reward += reward
            act = self.agent.take_action(obs_n, True)
            step += 1
        return episode_reward