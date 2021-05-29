__author__ = 'DafniAntotsiou'

'''
This script is heavily based on @openai/baselines.gail.mlp_policy.
'''

from baselines.common.mpi_running_mean_std import RunningMeanStd
import cat_dauggi.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype


class MlpPolicy(object):
    def __init__(self, name, *args, **kwargs):
        name = kwargs['prefix'] + name if 'prefix' in kwargs else name
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, prefix='', gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.prefix = prefix

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name=prefix + "ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter", reuse=tf.AUTO_REUSE):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol', reuse=tf.AUTO_REUSE):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name=prefix + "stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


class MlpPolicy_Custom(object):
    def __init__(self, name, *args, **kwargs):
        name = kwargs['prefix'] + name if 'prefix' in kwargs else name

        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, prefix='', gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.prefix = prefix

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name=prefix + "ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter", reuse=tf.AUTO_REUSE):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vff', reuse=tf.AUTO_REUSE):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            fc1 = tf.nn.tanh(tf.layers.dense(last_out, 256, name='fc1',
                                                      kernel_initializer=U.normc_initializer(1.0), bias_initializer=U.normc_initializer(0.01)))
            fc2 = tf.nn.tanh(tf.layers.dense(fc1, 128, name='fc2',
                                                  kernel_initializer=U.normc_initializer(1.0), bias_initializer=U.normc_initializer(0.01)))
            self.vpred = tf.layers.dense(fc2, 1, name='final', kernel_initializer=U.normc_initializer(1.0), bias_initializer=U.normc_initializer(0.01))[:,
                         0]

        with tf.variable_scope('pol', reuse=tf.AUTO_REUSE):
            last_out = obz
            fc1 = tf.nn.tanh(tf.layers.dense(last_out, 256, name='fc1',
                                                      kernel_initializer=U.normc_initializer(1.0), bias_initializer=U.normc_initializer(0.01)))
            fc2 = tf.nn.tanh(tf.layers.dense(fc1, 128, name='fc2',
                                                  kernel_initializer=U.normc_initializer(1.0), bias_initializer=U.normc_initializer(0.01)))

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(fc2, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=U.normc_initializer(0.01), bias_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(fc2, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01), bias_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name=prefix + "stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
