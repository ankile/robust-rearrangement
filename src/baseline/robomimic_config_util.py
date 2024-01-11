# from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/robomimic_config_util.py#L10

from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

def get_robomimic_config(
        algo_name='bc_rnn', 
        hdf5_type='low_dim', 
        task_name='square', 
        dataset_type='ph'
    ):
    base_dataset_dir = '/tmp/null'
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type, 
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
    config = algo_config_modifier(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type,
    )
    return config


def get_rm_config(algo_name="bc"):

    '''
    from https://github.com/ARISE-Initiative/robomimic/blob/master/examples/train_bc_rnn.py
    see also https://github.com/ARISE-Initiative/robomimic/blob/master/examples/simple_config.py
    for more minimal
    '''

    config = config_factory(algo_name=algo_name)

    # # fetch sequences of length 10 from dataset for RNN training
    # config.train.seq_length = 10

    # # observation encoder architecture - applies to all networks that take observation dicts as input
    # config.observation.encoder.rgb.core_class = "VisualCore"
    # config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    # config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'                         # ResNet backbone for image observations (unused if no image observations)
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False                # kwargs for visual core
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    # config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
    # config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32                      # Default arguments for "SpatialSoftmax"
    # config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
    # config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
    # config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0                  # Default arguments for "SpatialSoftmax"

    # ### Observation Config ###
    # config.observation.modalities.obs.low_dim = [               # specify low-dim observations for agent
    #     "proprio",
    # ]
    # config.observation.modalities.obs.rgb = [
    #     "rgb",
    #     "rgb_wrist",
    # ]

    # config.observation.modalities.obs.depth = [
    #     "depth",
    #     "depth_wrist",
    # ]
    # config.observation.modalities.obs.scan = [
    #     "scan",
    # ]
    # config.observation.modalities.goal.low_dim = []             # no low-dim goals
    # config.observation.modalities.goal.rgb = []               # no rgb image goals

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