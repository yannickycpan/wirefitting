import tensorflow as tf

def create_act_q_nn(scopename, stateDim, n_app_point, actionDim, actionBound,
                    n_hidden1, n_hidden2, dtype = tf.float32):
    with tf.variable_scope(scopename):
        state_input = tf.placeholder(dtype, [None, stateDim])
        state_hidden1 = tf.contrib.layers.fully_connected(state_input, n_hidden1)
        state_hidden2_acts = tf.contrib.layers.fully_connected(state_hidden1, n_hidden2)
        state_hidden2_qvals = tf.contrib.layers.fully_connected(state_hidden1, n_hidden2)

        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        interim_acts = tf.contrib.layers.fully_connected(state_hidden2_acts, n_app_point * actionDim,
                                                         activation_fn=tf.tanh,
                                                         weights_initializer=w_init) * actionBound
        interim_qvalues = tf.contrib.layers.fully_connected(state_hidden2_qvals, n_app_point,
                                                            activation_fn=None,
                                                            weights_initializer=w_init)
        # get best action and highest q value
        maxqvalue = tf.reduce_max(interim_qvalues, axis=1)
        maxind = tf.argmax(interim_qvalues, axis=1)
        rowinds = tf.range(0, tf.cast(tf.shape(state_input)[0], tf.int64), 1)
        maxind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(maxind, [-1, 1])], axis=1)
        bestacts = tf.gather_nd(tf.reshape(interim_acts, [-1, n_app_point, actionDim]), maxind_nd)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return state_input, interim_acts, interim_qvalues, maxqvalue, bestacts, tvars


def create_interpolation(scopename, interim_actions, interim_qvalues, max_q, n_app_point,
                         actionDim, smooth_eps = 0.000001, dtype = tf.float32):
    with tf.variable_scope(scopename):
        action_input = tf.placeholder(dtype, [None, actionDim])
        tiled_action_input = tf.tile(action_input, [1, n_app_point])
        reshaped_action_input = tf.reshape(tiled_action_input, [-1, n_app_point, actionDim])
        reshaped_action_output = tf.reshape(interim_actions, [-1, n_app_point, actionDim])
        # distance is b * n mat, n is number of points to do interpolation
        act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis=2)
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        #smooth_c = tf.sigmoid(tf.get_variable("smooth_c", [1, n_app_point], initializer=w_init, dtype=dtype))
        smooth_c = 1.0
        q_distance = smooth_c * (tf.reshape(max_q, [-1, 1]) - interim_qvalues)
        distance = act_distance + q_distance + smooth_eps
        weight = 1.0 / distance
        # weight sum is a matrix b*1, b is batch size
        weightsum = tf.reduce_sum(weight, axis=1, keep_dims=True)
        weight_final = weight / weightsum
        qvalue = tf.reduce_sum(tf.multiply(weight_final, interim_qvalues), axis=1)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
    return action_input, qvalue, tvars