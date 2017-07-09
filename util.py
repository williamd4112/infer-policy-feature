import tensorflow as tf

def get_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config
