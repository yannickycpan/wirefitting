from utils.replaybuffer import RecencyBuffer as recbuff
import numpy as np

class Agent(object):
    def __init__(self, params):
        #print('params are: --------------------------------------------------- ', params)
        self.name = params['name']
        self.replaybuffer = recbuff(params['bufferSize'])
        self.batchSize = params['batchSize']
        self.updateFrequency = params['updateFrequency'] if 'updateFrequency' in params else 1
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilonDecay']
        self.epsilon_min = params['epsilonMin']
        self.gamma = params['gamma']

        self.saveModel = params['saveModel']
        self.useSavedModel = params['useSavedModel']
        self.notTrain = params['notTrain']

        self.warm_up_steps = params['warmUpSteps']

        self.actionDim = params['actionDim']
        self.actionBound = params['actionBound']
        #print('update frequency is :: ', self.updateFrequency)

        self.n_episode = 0.0
        self.n_samples = 0
        self.start_learning = False

        self.agent_function = None

    def take_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon or not self.start_learning:
            action = np.random.randint(self.actionDim)
        else:
            action = self.agent_function.take_action(state)
        if self.start_learning:
            self.epsilon = self.epsilon_min if self.epsilon < self.epsilon_min else self.epsilon * self.epsilon_decay
        return action

    def take_action_eval(self, state):
        action = self.agent_function.take_action(state)
        if self.actionBound is not None:
            return np.clip(action, -self.actionBound, self.actionBound)
        else:
            return action

    def update(self, s, a, sp, r, episodeEnd):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        #if episodeEnd:
            #print('current quantile value is ::::::::::::::::::::: ', self.agent_function.compute_max_quantile_values(s))
        #    print('acted are: ', a, self.agent_function.compute_acted(s, a))
        self.replaybuffer.add(s, a, sp, r, gamma)
        self.n_samples += 1
        if len(self.replaybuffer.buffer) >= self.warm_up_steps and self.n_samples % self.updateFrequency == 0:
            bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
            self.agent_function.train(bs, ba, bsp, br, bgamma)
            self.start_learning = True
            #print('-------------------it is training--------------------', self.n_samples)

    def save_model(self, env_name, agent_name):
        if self.saveModel:
            self.agent_function.save_model(env_name, agent_name)
        return None

    def episode_reset(self):
        return