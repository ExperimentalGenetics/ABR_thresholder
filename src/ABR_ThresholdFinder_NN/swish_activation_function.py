# Tensorflow and tf.keras
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K


# define activation function
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    return K.sigmoid(x) * x


utils.get_custom_objects().update({'swish': swish})
