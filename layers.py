import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs
from layers_lib import lstm_layer,lstm_cond_layer,lstm_softatt_layer,lstm_cond_softatt_layer

class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('lstm_layer.param_init', 'lstm_layer.get_layer'),
            'lstm_cond': ('lstm_cond_layer.param_init', 'lstm_cond_layer.get_layer'),
            'lstm_softatt': ('lstm_softatt_layer.param_init', 'lstm_softatt_layer.get_layer'),
            'lstm_cond_softatt': ('lstm_cond_softatt_layer.param_init', 'lstm_cond_softatt_layer.get_layer'),            
            }
        self.rng_numpy, self.rng_theano = get_two_rngs()

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return eval(fns[0]), eval(fns[1])

    def dropout_layer(self, state_before, use_noise, trng):

        proj = T.switch(use_noise,
                             state_before *
                             trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                             state_before * 0.5)
        return proj

    def fflayer(self, tparams, state_below, activ='lambda x: T.tanh(x)', prefix='ff', **kwargs):

        return eval(activ)(T.dot(state_below, tparams[_p(prefix,'W')])+
                           tparams[_p(prefix,'b')])

    def param_init_fflayer(self, params, nin, nout, prefix=None):
        assert prefix is not None
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
        return params
