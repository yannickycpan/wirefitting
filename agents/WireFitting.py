import os
import sys
import numpy as np
import tensorflow as tf
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from FunctionApproximator import FunctionApproximator
from network.wirefitting_network import create_act_q_nn, create_interpolation
from network.operations import update_target_nn_assign, update_target_nn_move
from Agent import Agent


class wirefittingnn(FunctionApproximator):
    def __init__(self, params):
        super(wirefittingnn, self).__init__(params)

        self.app_points = params['numControllers']
        self.n = 1
        
        with self.g.as_default():
            self.is_training = tf.placeholder(tf.bool, [])
            self.state_input, self.interim_actions, self.interim_qvalues, self.max_q, self.bestact, self.tvars_in_actnn \
                = create_act_q_nn("actqNN", self.stateDim, self.app_points,
                                  self.actionDim, self.actionBound, self.n_h1, self.n_h2)

            self.tar_state_input, self.tar_interim_actions, self.tar_interim_qvalues, self.tar_max_q, self.tar_bestact, self.tar_tvars_in_actnn \
                = create_act_q_nn("target_actqNN", self.stateDim, self.app_points,
                                  self.actionDim, self.actionBound, self.n_h1, self.n_h2)

            self.action_input, self.qvalue, self.tvars_interplt \
                = create_interpolation("interpolation", self.interim_actions, self.interim_qvalues,
                                       self.max_q, self.app_points, self.actionDim)

            self.tar_action_input, self.tar_qvalue, self.tar_tvars_interplt \
                = create_interpolation("target_interpolation", self.tar_interim_actions,
                                       self.tar_interim_qvalues, self.tar_max_q, self.app_points, self.actionDim)
            #one list includes all vars
            self.tvars = self.tvars_in_actnn + self.tvars_interplt
            self.tar_tvars = self.tar_tvars_in_actnn + self.tar_tvars_interplt
            #define loss operation
            self.qtarget_input, self.interplt_loss = self.define_loss("losses")

            self.params_update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.interplt_loss)
            #update target network
            self.init_target = update_target_nn_assign(self.tar_tvars, self.tvars)
            self.update_target = update_target_nn_move(self.tar_tvars, self.tvars, self.tau)
            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

    def define_loss(self, scopename):
        with tf.variable_scope(scopename):
            qtargets = tf.placeholder(self.dtype, [None])
            interplt_loss = tf.losses.mean_squared_error(qtargets, self.qvalue)
        return qtargets, interplt_loss


    def take_action(self, state):
        bestact = self.sess.run(self.bestact, {self.state_input: state.reshape(-1, self.stateDim)})
        return bestact.reshape(-1)


    def computeQtargets(self, state_tp, reward, gamma):
        Sp_qmax = self.sess.run(self.tar_max_q, {self.tar_state_input: state_tp})
        qTargets = reward + gamma*np.squeeze(Sp_qmax)
        return qTargets


    def train(self, state, action, state_tp, reward, gamma):
        qtargets = self.computeQtargets(state_tp, reward, gamma)
        self.sess.run(self.params_update, feed_dict = {self.state_input: state.reshape(-1, self.stateDim),
                                                       self.action_input: action.reshape(-1, self.actionDim),
                                                       self.qtarget_input: qtargets.reshape(-1)})
        self.sess.run(self.update_target)


class WireFittingAgent(Agent):
    def __init__(self, params):
        super(WireFittingAgent, self).__init__(params)

        self.agent_function = wirefittingnn(params)
        self.notTrain = False
        self.noise_t = np.zeros(self.actionDim)
        if params['useSavedModel']:
            self.agent_function.restore(params['modelPath'])
            self.notTrain = True
            self.start_learning = True

    def take_action(self, state):
        if not self.start_learning:
            return np.random.uniform(-self.actionBound, self.actionBound, self.actionDim)
        action = self.agent_function.take_action(state)
        self.noise_t += np.random.normal(np.zeros(self.actionDim),
                                             0.2 * np.ones(self.actionDim)) - self.noise_t * 0.15
        action = action + self.noise_t
        return np.clip(action, -self.actionBound, self.actionBound)

    '''here we use and store option, not primal actions, so primal action is not directly used for training'''
    '''they passed a in this function is the primal action, not option'''
    def update(self, s, a, sp, r, episodeEnd):
        if self.notTrain:
            return
        gamma = self.gamma if not episodeEnd else 0.0
        if episodeEnd:
            self.n_episode += 1.
        self.replaybuffer.add(s, a, sp, r, gamma)
        if len(self.replaybuffer.buffer) >= self.warm_up_steps:
            bs, ba, bsp, br, bgamma = self.replaybuffer.sample_batch(self.batchSize)
            self.agent_function.train(bs, ba, bsp, br, bgamma)
            self.start_learning = True
