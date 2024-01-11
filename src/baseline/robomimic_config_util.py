# from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/robomimic_config_util.py#L10
from robomimic.config import config_factory


def get_rm_config(algo_name="bc"):

    '''
    from https://github.com/ARISE-Initiative/robomimic/blob/master/examples/train_bc_rnn.py
    see also https://github.com/ARISE-Initiative/robomimic/blob/master/examples/simple_config.py
    for more minimal
    '''

    config = config_factory(algo_name=algo_name)

    ### Algo Config ###

    # MLP network architecture (layers after observation encoder and RNN, if present)
    config.algo.actor_layer_dims = (300, 400)           # MLP layers between RNN layer and action output

    # # stochastic GMM policy
    # config.algo.gmm.enabled = True                      # enable GMM policy - policy outputs GMM action distribution
    # config.algo.gmm.num_modes = 5                       # number of GMM modes
    # config.algo.gmm.min_std = 0.01                      # minimum std output from network
    # config.algo.gmm.std_activation = "softplus"         # activation to use for std output from policy net
    # config.algo.gmm.low_noise_eval = True               # low-std at test-time

    # rnn policy config
    config.algo.rnn.enabled = True      # enable RNN policy
    config.algo.rnn.horizon = 10        # unroll length for RNN - should usually match train.seq_length
    config.algo.rnn.hidden_dim = 1200   # hidden dimension size
    config.algo.rnn.rnn_type = "LSTM"   # rnn type - one of "LSTM" or "GRU"
    config.algo.rnn.num_layers = 2      # number of RNN layers that are stacked
    config.algo.rnn.open_loop = False   # if True, action predictions are only based on a single observation (not sequence) + hidden state
    config.algo.rnn.kwargs.bidirectional = False          # rnn kwargs

    return config