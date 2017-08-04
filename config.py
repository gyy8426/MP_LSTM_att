from jobman import DD

RAB_DATASET_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/predatas/'
RAB_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/Resnet/'
OPT_FEATURE_BASE_PATH = '/mnt/data3/guoyuyu/datasets/MSVD/features/OptFlow_Resnet/'
RAB_EXP_PATH = '/home/guoyuyu/results/youtube//MP_lstm_condsoftatt_res_initff_ctx_Nopre_1/'

config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': True,
    'attention': DD({
        'reload_': False,
        'verbose': True,
        'debug': False,
        'save_model_dir': RAB_EXP_PATH + 'save_dir/',
        'from_dir': RAB_EXP_PATH + 'from_dir/',
        # dataset
        'dataset': 'youtube2text',
        'video_feature': 'googlenet',
        'K':28, # 26 when compare
        'OutOf':None,
        # network
        'dim_word':512,#468, # 474
        'tu_dim': 512,
        'mu_dim': 1024,
        'ctx_dim':-1,# auto set
        'att_type':'lstm_cond_softatt',
        'n_layers_out':1, # for predicting next word
        'n_layers_init':0,
        'encoder_dim': 1024,#300,
        'init_tulstm':True,
        'prev2out':False,
        'ctx2out':True,
        'selector':True,
        'n_words':20000,
        'maxlen':30, # max length of the descprition
        'use_dropout':True,
        'isGlobal': True,
        # training
        'patience':20,
        'max_epochs':500,
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.0001,
        'optimizer':'adadelta',
        'clip_c': 10.,
        # minibatches
        'batch_size': 64, # for trees use 25
        'valid_batch_size':200,
        'dispFreq':10,
        'validFreq':1000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        }),
    })
