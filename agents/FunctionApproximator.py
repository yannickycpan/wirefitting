import tensorflow as tf
import datetime
import os


class FunctionApproximator(object):
    def __init__(self, params):
        self.learning_rate = params['alpha']

        self.n_h1 = params['n_h1']
        self.n_h2 = params['n_h2']

        # target network moving rate
        self.tau = params['tau']

        self.dtype = tf.float32
        self.stateDim = params['stateDim']
        self.actionDim = params['actionDim']
        self.actionBound = params['actionBound']

        self.use_atari_nn = params['type'] if 'type' in params else False

        self.g = tf.Graph()

        # this variable needs to be initialized in subclass
        self.saver = None

        with self.g.as_default():
            tf.set_random_seed(params['seed'])
            self.sess = tf.Session()

    def compute_action_value(self, scopename):
        return

    def define_loss(self, scopename):
        return

    def train(self, s, a, sp, r, gamma):
        return

    def take_action(self, state):
        return

    def save_model(self, env_name, agent_name):
        with self.g.as_default():
            nt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            name = env_name + agent_name + nt
            filepath = name + '/nnmodel.ckpt'
            if not os.path.exists(name):
                os.makedirs(name)
            '''init saver'''
            savepath = self.saver.save(self.sess, filepath)
            print('model is saved at %s ' % savepath)

    def restore(self, file_path):
        tf.reset_default_graph()
        '''init saver'''
        self.saver.restore(self.sess, file_path)
        print('model restored !!!!!!')