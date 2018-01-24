"""
Defines agents which use the tensorforce library
"""
import abc

from tensorforce.agents import DQNAgent
from tensorforce.core.networks import Layer
import tensorforce as tf
from tensorforce.core.preprocessing import Preprocessor

class Conv2DPreprocessor(Preprocessor):
    """
    Adds the channel dimension at the front
    """

    def __init__(self, scope='sequence', summary_labels=()):
        super(Conv2DPreprocessor, self).__init__(scope=scope, summary_labels=summary_labels)

    def reset(self):
        pass

    def tf_process(self, tensor):
        return tf.expand_dims(tensor, 1)

    def processed_shape(self, shape):
        return (1,) + shape

# TODO: This doesn't work because UPDATE_OPS needs to be put in the TF graph.
class BatchNormalization(Layer):

    def __init__(self, variance_epsilon=1e-6, scope='batchnorm', summary_labels=None):
        super(BatchNormalization, self).__init__(scope=scope, summary_labels=summary_labels)
        self.variance_epsilon = variance_epsilon

    def tf_apply(self, x, update):
        mean, variance = tf.nn.moments(x, axes=tuple(range(x.shape.ndims - 1)))
        return tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=None,
            scale=None,
            variance_epsilon=self.variance_epsilon
        )


class Permute(Layer):
    def __init__(self, dims, scope='permute', summary_labels=None):
        super(Permute, self).__init__(scope=scope, summary_labels=summary_labels)
        self.dims = tuple(dims)

    def tf_apply(self, x, update):
        return tf.transpose(x, perm=self.dims)

class TensorforceRLAgent:
    """
    Abstract class for an agent that can be trained on an environment
    """

    def __init__(self, env):
        """
        :param Environment env: gym environment the agent will act in
        """
        self.env = env

    @abc.abstractmethod
    def agent(self):
        """
        create and initialize the tensorforce agent

        :return Agent: the fully initialized tensorforce agent.
        """
        pass

class TensorforceDQNAgent(TensorforceRLAgent):

    def agent(self):
        nb_actions = self.env.action_space.n
        obs_dim = len(self.env.observation_space.spaces)
        obs_dim_2 = self.env.observation_space.spaces[0].shape
        return DQNAgent(
            states=dict(type='float32', shape=(obs_dim, obs_dim_2)),
            actions=dict(type='int', num_actions=nb_actions),
            network=[
                dict(type=Permute, dims=(0, 2, 3, 1)),
                dict(type='conv2d', size=5, window=(obs_dim, 1), stride=(obs_dim, 1),
                     padding='SAME', l2_regularization=1e-4),
                dict(type=BatchNormalization),
                dict(type=Permute, dims=(0, 3, 2, 1)),
                dict(type='conv2d', size=5, window=(5, 1), stride=(5, 1),
                     padding='SAME', l2_regularization=1e-4),
                dict(type=BatchNormalization),
                dict(type=Permute, dims=(0, 3, 2, 1)),
                dict(type='conv2d', size=5, window=(5, 1), stride=(5, 1),
                     padding='SAME', l2_regularization=1e-4),
                dict(type=BatchNormalization),
                dict(type=Permute, dims=(0, 3, 2, 1)),
                dict(type='conv2d', size=5, window=(5, 1), stride=(5, 1),
                     padding='SAME', l2_regularization=1e-4),
                dict(type=BatchNormalization),
                dict(type='flatten'),
                dict(type='dense',size=nb_actions,activation='softmax')
            ],
            states_preprocessing=[dict(type=Conv2DPreprocessor)]
)





