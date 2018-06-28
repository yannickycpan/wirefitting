import gym
import sys
import numpy as np
try:
    import roboschool
except ImportError:
    print('Running without using roboschool')
try:
    import utils.gymbaselinecommon as gymbase
except ImportError:
    print('Running without using gymbaselinecommon')

#This file provide environments to interact with, consider actions as continuous, need to rewrite otherwise
class Environment(object):
    def __init__(self, env_params):
        self.name = env_params['name']
        self.instance = gym.make(env_params['name'])
        self.instance.seed = env_params['seed']
        # maximum number of steps allowed for each episode
        #self.TOTAL_STEPS_LIMIT = env_params['TotalSamples']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']

        #self.instance._max_episode_steps = env_params['EpisodeSamples']
        # state info
        self.stateDim = self.getStateDim()
        self.stateRange = self.getStateRange()
        self.stateMin = self.getStateMin()
        self.stateBounded = env_params['stateBounded']
        #self.stateBounded = False if np.any(np.isinf(self.instance.observation_space.high)) or np.any(np.isinf(self.instance.observation_space.low)) else True
        # action info
        self.actionDim = self.getControlDim()
        self.actionBound = self.getActBound()
        self.actMin = self.getActMin()

        #DEBUG
        print('stateDim:',self.stateDim)
        # print('stateRange:', self.stateRange)
        # print('stateMin:', self.stateMin)
        print("stateBounded :: ", self.stateBounded)
        print("actionDim", self.actionDim)
        # print('actRange', self.actRange)
        print("actionBound :: ", self.actionBound)
        # print('actMin', self.actMin)
        # exit()
        
    # Reset the environment for a new episode. return the initial state
    def reset(self):
        state = self.instance.reset()
        if self.stateBounded:
            # normalize to [-1,1]
            scaled_state = 2.*(state - self.stateMin)/self.stateRange - 1.
            return scaled_state
        return np.array(state)

    def step(self, action):
        state, reward, done, info = self.instance.step(action)
        #self.instance.render()
        if self.stateBounded:
            scaled_state = 2.*(state - self.stateMin)/self.stateRange - 1.
            return (scaled_state, reward, done, info)
        return (np.array(state), reward, done, info)

    def getStateDim(self):
        dim = self.instance.observation_space.shape
        if len(dim) < 2:
            return dim[0]
        return dim
  
    # this will be the output units in NN
    def getControlDim(self):
        # if discrete action
        if hasattr(self.instance.action_space, 'n'):
            return int(self.instance.action_space.n)
        # if continuous action
        return int(self.instance.action_space.sample().shape[0])

    # Return action ranges
    def getActBound(self):
        #print self.instance.action_space.dtype
        if hasattr(self.instance.action_space, 'high'):
            #self.action_space = spaces.Box(low=self.instance.action_space.low, high=self.instance.action_space.high, shape=self.instance.action_space.low.shape, dtype = np.float64)
            return self.instance.action_space.high[0]
        return None

    # Return action min
    def getActMin(self):
        if hasattr(self.instance.action_space, 'low'):
            return self.instance.action_space.low
        return None

    # Return state range
    def getStateRange(self):
        return self.instance.observation_space.high - self.instance.observation_space.low
    
    # Return state min
    def getStateMin(self):
        return self.instance.observation_space.low

    # Close the environment and clear memory
    def close(self):
        self.instance.close()


class AtariEnvironment(Environment):
    def __init__(self, env_params):
        self.instance = gym.make(env_params['name'])
        if 'gymbase' in sys.modules:
            self.instance = gymbase.NoopResetEnv(self.instance, noop_max=30)
            #self.instance = gymbase.MaxAndSkipEnv(self.instance, skip=4)
            self.instance = gymbase.wrap_deepmind(self.instance, episode_life=False,
                                              clip_rewards=False, frame_stack=True,
                                              scale=True)
        self.instance._max_episode_steps = 4000
        self.instance.seed = env_params['seed']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']

        # self.instance._max_episode_steps = env_params['EpisodeSamples']
        # state info
        self.stateDim = self.getStateDim()
        self.stateRange = self.getStateRange()
        self.stateMin = self.getStateMin()
        self.stateBounded = env_params['stateBounded']
        # self.stateBounded = False if np.any(np.isinf(self.instance.observation_space.high)) or np.any(np.isinf(self.instance.observation_space.low)) else True
        # action info
        self.actionDim = self.getControlDim()
        self.actionBound = self.getActBound()
        self.actMin = self.getActMin()

        print('stateDim:', self.stateDim)
        print('stateRange:', self.stateRange)
        print('stateMin:', self.stateMin)
        print("stateBounded :: ", self.stateBounded)
        print("actionDim", self.actionDim)
        print("actionBound :: ", self.actionBound)
        print('actMin', self.actMin)


'''this environment is NOT tested!!!'''
class BlockWorld(Environment):
    def __init__(self, env_params):
        # maximum number of steps allowed for each episode
        # block world with two walls, space is [0,1]^2
        #self.TOTAL_STEPS_LIMIT = env_params['TotalSamples']
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSamples']
        self.TOTAL_EPISODES = env_params['NumEpisodes']
        # state info
        self.stateDim = 2
        self.stateRange = np.array([1.0, 1.0])
        self.stateMin = np.array([0.0, 0.0])
        self.stateBounded = True        # action info
        self.actionDim = 2
        self.actRange = np.array([1.0, 1.0])
        self.actionBound = np.array([1.0, 1.0])
        self.actMin = np.array([-1.0, -1.0])
        self.stepsize = 0.05
        #define right top point
        self.rightop = np.array([1., 1.])
        #goal area is the square area
        self.goalwidth = np.array([0.05, 0.05])
        #define wall lefttop location, wall width
        self.wall_width = 0.2
        self.hole_lefttop_1 = np.array([0.3, 0.8])
        self.hole_lefttop_2 = np.array([0.7, 0.4])
        self.hole_length = 0.2

    #every time start from the left top corner
    def reset(self):
        self.state = np.array([0.0, 1.0])
        return self.state
    
    def reachGoal(self, nextstate):
        if nextstate[0] > self.rightop[0] - self.goalwidth and nextstate[1] > self.rightop[1] - self.goalwidth:
            return True
        return False
    
    def getReward(self, action, nextstate):
        if nextstate[0] > self.rightop[0] - self.goalwidth and nextstate[1] > self.rightop[1] - self.goalwidth:
            return 0.
        return -1
    
    def hit_one_wall(self, nextstate, holelefttop, wall_width, hole_length):
        if nextstate[0] > holelefttop[0] and nextstate[0] < holelefttop[0] + wall_width:
            if nextstate[1] > holelefttop[1] - hole_length and nextstate[1] < holelefttop[1]:
                return False
            else:
                return True
        return False
    
    def hitwall(self, nextstate):
        wall1 = self.hit_one_wall(nextstate, self.hole_lefttop_1, self.wall_width, self.hole_length)
        wall2 = self.hit_one_wall(nextstate, self.hole_lefttop_2, self.wall_width, self.hole_length)
        return wall1 and wall2

    def step(self, action):
        actual_step = action*self.stepsize + np.random.normal(0.0, 0.005, size = 2)
        nextstate = self.state + actual_step
        reachgoal = self.reachGoal(nextstate)
        hitwall = self.hitwall(nextstate)
        if hitwall:
            nextstate = self.state
        else:
            self.state = nextstate
        done = True if reachgoal else False
        reward = self.getReward(action, nextstate)
        return self.state, reward, done, None
