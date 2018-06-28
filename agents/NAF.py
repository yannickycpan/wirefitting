# from agents.agents import Agent # for python3
from agents import Agent # for python2
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque

from utils.replaybuffer import ReplayBuffer

class nafnn:
    def __init__(self, env, params, random_seed):
        
        self.lc = params['learning_rate']
    
        self.n_h1 = 200
        self.n_h2 = 200
        
        self.tau = params['tau']
        
        #self.use_doubleQ = True
        
        self.decay_rate = params['lc_decay_rate']
        self.decay_step = params['lc_decay_step']
        
        print 'decay step, decay rate:: ', self.decay_rate, self.decay_step

        self.stateDim = env.stateDim # 2
        self.actionDim  = env.actionDim # 1
        self.actionBound = env.actionBound[0]
        
        #record step n
        self.n = 0

        self.dtype = tf.float64

        self.g = tf.Graph()
        
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.is_training = tf.placeholder(tf.bool, [])
            self.state_input, self.action_input, self.qvalue, self.max_q, self.bestact, self.Lmat_columns, self.tvars = self.create_q_nn("QNN")
            self.tar_state_input, self.tar_action_input, self.tar_qvalue, self.tar_max_q, self.tar_bestact, _, self.tar_tvars = self.create_q_nn("target_QNN")
            
            #define loss operation
            self.qtarget_input, self.qloss = self.define_loss("nafloss")

            #define optimization
            #self.global_step = tf.Variable(0, trainable=False)
            #learning_rate = tf.train.exponential_decay(self.lc, self.global_step, self.decay_step, self.decay_rate, staircase=True)
            self.params_update = tf.train.AdamOptimizer(self.lc).minimize(self.qloss)
            #self.step_add = tf.assign_add(self.global_step, 1)
            #update target network
            self.init_target = [tf.assign(self.tar_tvars[idx], self.tvars[idx]) for idx in range(len(self.tar_tvars))]
            self.update_target = [tf.assign_add(self.tar_tvars[idx], self.tau*(self.tvars[idx] - self.tar_tvars[idx])) for idx in range(len(self.tvars))]
            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

    def define_loss(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                qtargets = tf.placeholder(self.dtype, [None])
                print 'qvalue shape is :: ', self.qvalue.shape
                loss = tf.reduce_sum(tf.square(qtargets - tf.squeeze(self.qvalue)))
        return qtargets, loss

    def create_q_nn(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                state_input = tf.placeholder(self.dtype, [None, self.stateDim])
                action_input = tf.placeholder(self.dtype, [None, self.actionDim])
                #three branch: first output action
                action_hidden1 = slim.fully_connected(state_input, self.n_h1, activation_fn = tf.nn.relu)
                action_hidden2 = slim.fully_connected(action_hidden1, self.n_h2, activation_fn = tf.nn.relu)
                action = slim.fully_connected(action_hidden2, self.actionDim, activation_fn = tf.nn.tanh)*self.actionBound
                #value branch
                value_hidden1 = slim.fully_connected(state_input, self.n_h1, activation_fn = tf.nn.relu)
                value_hidden2 = slim.fully_connected(value_hidden1, self.n_h2, activation_fn = tf.nn.relu)
                w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                value = slim.fully_connected(value_hidden2, 1, activation_fn = None, weights_initializer=w_init)
                #Lmat branch
                lmat_hidden1 = slim.fully_connected(state_input, self.n_h1, activation_fn = tf.nn.relu)
                lmat_hidden2 = slim.fully_connected(lmat_hidden1, self.n_h2, activation_fn = tf.nn.relu)
                act_mu_diff = action_input - action
                #Lmat_flattened = slim.fully_connected(state_hidden1, (1+self.actionDim)*self.actionDim/2, activation_fn = None)
                Lmat_diag = [tf.exp(slim.fully_connected(lmat_hidden2, 1, activation_fn = None)) for _ in range(self.actionDim)]
                Lmat_nondiag = [slim.fully_connected(lmat_hidden2, k-1, activation_fn = None) for k in range(self.actionDim, 1, -1)]
                #in Lmat_columns, if actdim = 1, first part is empty
                Lmat_columns = [tf.concat((Lmat_diag[id], Lmat_nondiag[id]),axis=1) for id in range(len(Lmat_nondiag))] + [Lmat_diag[-1]]
                act_mu_diff_Lmat_prod = [tf.reduce_sum(tf.slice(act_mu_diff,[0,cid],[-1,-1])*Lmat_columns[cid], axis=1, keep_dims=True) for cid in range(len(Lmat_columns))]
                #prod_tensor should be dim: batchsize*actionDim
                prod_tensor = tf.concat(act_mu_diff_Lmat_prod, axis = 1)
                print 'prod tensor shape is :: ', prod_tensor.shape
                adv_value = -0.5*tf.reduce_sum(prod_tensor*prod_tensor, axis = 1, keep_dims=True)
                q_value = value + adv_value
                max_q = value
                #get variables
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return state_input, action_input, q_value, max_q, action, Lmat_columns, tvars

    '''return an action to take for each state, NOTE this action is in [-1, 1]'''
    def takeAction(self, state):
        bestact, Lmat_columns = self.sess.run([self.bestact,self.Lmat_columns], {self.state_input: state.reshape(-1, self.stateDim), self.is_training: False})
        Lmat = np.zeros((self.actionDim, self.actionDim))
        #print 'the Lmat columns are --------------------------------------------- '
        for i in range(self.actionDim):
            Lmat[i:,i] = np.squeeze(Lmat_columns[i])
        covmat = np.linalg.inv(Lmat.dot(Lmat.T))
        #print 'sampled act is ------------------ ', sampled_act
        return bestact.reshape(-1), covmat

    # similar to takeAction(), except this is for targets, returns QVal instead, and calculates in batches
    def computeQtargets(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            Sp_qmax = self.sess.run(self.tar_max_q, {self.tar_state_input: state_tp, self.is_training: False})
            # this is a double Q DDQN learning rule
            #Sp_bestacts = self.sess.run(self.bestact, {self.state_input: state_tp})
            #Sp_qmax = self.sess.run(self.tar_qvalue, {self.tar_state_input: state_tp, self.tar_action_input: Sp_bestacts})
            qTargets = reward + gamma*np.squeeze(Sp_qmax)
            return qTargets
    
    def update_vars(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            qtargets = self.computeQtargets(state, action, state_tp, reward, gamma)
            #self.sess.run(self.step_add)
            self.sess.run(self.params_update, feed_dict = {self.state_input: state.reshape(-1, self.stateDim), self.action_input: action.reshape(-1, self.actionDim), self.qtarget_input: qtargets.reshape(-1), self.is_training: True})
            self.sess.run(self.update_target)
        #print gdstep
        return None

    def reset(self):
        with self.g.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

class NAF(Agent):
    def __init__(self, env, params, random_seed):
        super(NAF, self).__init__(env)

        np.random.seed(random_seed) # Random action selection
        random.seed(random_seed) # Experience Replay Buffer

        self.noise_scale = params['noise_scale']
        self.epsilon = params['epsilon'] # 0.3
        self.epsilon_decay = params['epsilon_decay'] # 0.9
        self.epsilon_decay_step = params['epsilon_decay_step'] # 100
        self.policyfunc = nafnn(env, params, random_seed)

        self.replay_buffer = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']

        self.gamma = params['gamma'] # 0.99

        self.warmup_steps = params['warmup_steps'] 

        # self.noise_t = np.zeros(self.actionDim)
        self.action_is_greedy = None
        self.eps_decay = True
        
        self.cum_steps = 0 # cumulative steps across episodes
    
        #print('agent params gamma, epsilon', self.gamma, self.epsilon)


    def update(self, S, Sp, r, a, episodeEnd):
        if not episodeEnd:
            self.replay_buffer.add(S, a, r, Sp, self.gamma)
            self.learn()
        else:
            self.replay_buffer.add(S, a, r, Sp, 0.0)
            self.learn()
    
    def learn(self):
        if self.replay_buffer.getSize() > max(self.warmup_steps, self.batch_size):
            s, a, r, sp, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.policyfunc.update_vars(s, a, sp, r, gamma)
        else:
            return
        #print r
        #self.policyfunc.performtest(s, a, sp, r, gamma)

    def takeAction(self, state, isTrain):
        # epsilon greedy
        meanact, covmat = self.policyfunc.takeAction(state)
        #print bestact
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.actionMin, self.actionMax, self.actionDim)
        #action = self.env.instance.action_space.sample()
        else:
            if isTrain:
                action = np.random.multivariate_normal(meanact, self.noise_scale*covmat)
            #print self.noise_scale*covmat
            else:
                action = meanact
        self.cum_steps +=1
        #print self.actionMin, self.actionMax
        return np.clip(action, self.actionMin[0], self.actionMax[0])

    def getAction(self, state, isTrain):
        self.next_action = self.takeAction(state, isTrain)
        return self.next_action, self.action_is_greedy
    
    def start(self, state, isTrain):
        self.next_action = self.takeAction(state, isTrain)
        return self.next_action

    def reset(self):
        # self.erbuffer = [] # maybe do not reset erbuffer
        self.noise_t = np.zeros(self.actionDim)
        self.action_is_greedy = None
        # self.policyfunc.reset() # This shouldn't be reset!
