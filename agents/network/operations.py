import tensorflow as tf


def update_target_nn_move(tar_tvars, tvars, tau):
    target_params_update = [tf.assign_add(tar_tvars[idx], tau * (tvars[idx] - tar_tvars[idx]))
                                 for idx in range(len(tvars))]
    return target_params_update

def update_target_nn_assign(tar_tvars, tvars):
    target_params_update = [tf.assign(tar_tvars[idx],  tvars[idx]) for idx in range(len(tvars))]
    return target_params_update