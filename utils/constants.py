###
# Mapping between configuration file that enables mutations and mutation classes
###
mutation_class_map = {
    "Mutation": "Mutation",
    "change_label": "ChangeLabelTDMut",
    "delete_training_data": "DeleteTDMut",
    "unbalance_train_data": "UnbalanceTDMut",
    "add_noise": "AddNoiseTDMut",
    "make_output_classes_overlap": "OutputClassesOverlapTDMUT",
    ####
    "change_batch_size": "ChangeBatchSizeHPMut",
    "change_epochs": "ChangeEpochsHPMut",
    "change_learning_rate": "ChangeLearnRateHPMut",
    "disable_batching": "DisableBatchingHPMut",
    ####
    "change_activation_function": "ChangeActivationAFMut",
    "remove_activation_function": "RemoveActivationAFMut",
    "add_activation_function": "AddActivationAFMut",
    ####
    "change_weights_initialisation": "ChangeWeightsInitialisation",
    ####
    "change_optimisation_function": "ChangeOptimisationFunction",
    "change_gradient_clip": "ChangeGradientClip",
    ###
    "remove_validation_set": "RemoveValidationSet",
    ###
    "change_earlystopping_patience": "ChangeEarlyStoppingPatience",
    ###
    "add_bias": "AddBiasMut",
    "remove_bias": "RemoveBiasMut",
    ###
    "change_loss_function": "ChangeLossFunction",
    ###
    "change_dropout_rate": "ChangeDropoutRate",
    ###
    "add_weights_regularisation": "AddWeightsRegularisation",
    "change_weights_regularisation": "ChangeWeightsRegularisation",
    "remove_weights_regularisation": "RemoveWeightsRegularisation"
}

###
# Paths to save the models
###
save_paths = {
    "trained": "trained_models",
    "mutated": "mutated_models",
    "prepared": "prepared_models"
}


###
# Dict of imports of mutation operators
# Deprecated #TODO:remove
###
mutation_imports = {
    "D": "training_data_operators",
    "H": "hyperparams_operators"
}


###
# List of available activation functions (Keras)
# https://keras.io/activations/
###
activation_functions = [
    "elu",
    "softmax",
    "selu",
    "softplus",
    "softsign",
    "relu",
    #"tanh", 18.01 juan
    #"sigmoid", 18.01
    #"hard_sigmoid", 18.01
    #"exponential", 18.01
    #"linear" 18.01
]

#Mapping
#linear - softmax

###
# Operators lib
###
# operator_lib = "operators"
operator_mod = "operators"
operator_lib = ["activation_function_operators",
                "training_data_operators",
                "bias_operators",
                "weights_operators",
                "optimiser_operators",
                "dropout_operators,"
                "hyperparams_operators",
                "training_process_operators",
                "loss_operators"]

###
# Default number of runs
###
runs_number_default = 3 # 11.01 c 10 -> 3

###
# Binary search level of precision
###
binary_search_precision = 5

###
# Mutation params abbreviations
###
mutation_params_abbrvs = [
    "pct",
    "lbl",
    "optimisation_function_udp",
    "activation_function_udp",
    #"current_index",
    "loss_function_udp",
    "batch_size",
    "weights_initialisation_udp",
    "weights_regularisation_udp",
    "rate"
]

###
# Keras Optimisers and their default params
###

# List of Optimisers
keras_optimisers = [
    "sgd",
    "rmsprop",
    #"adagrad",
    #"adadelta",
    "adam",
    "adamax",
    "nadam"
]

# Dicts of default parameters

# SGD
sgd = {
    "learning_rate": 0.01,
    "momentum": 0.0,
    "Nesterov": False
}

# RMSprop
rmsprop = {
    "learning_rate": 0.001,
    "rho": 0.9}

# Adagrad
adagrad = {
    "learning_rate": 0.01
}

# Adadelta
adadelta = {
    "learning_rate": 1.0,
    "rho": 0.95
}

# Adam
adam = {
    "learning_rate":0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "amsgrad": False
}

# Adamax
adamax = {
    "learning_rate": 0.002,
    "beta_1": 0.9,
    "beta_2": 0.999
}

# Nadam
nadam = {
    "learning_rate": 0.002,
    "beta_1": 0.9,
    "beta_2": 0.999
}

###
# Keras Batch Sizes: In Reality Should be calculated automatically
###
batch_sizes = [
   32, 64, 256, 512
]

###
# Dropout Values: In Reality Should be calculated automatically
###
dropout_values = [
   0.125, 0.25, 0.75, 1.0
]


###
# Keras Weight Initialisers
###

keras_initialisers = [
    "zeros",
    "ones",
    "constant",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    #"variance_scaling",
    #"orthogonal", 18.01 cout
    #"identity",
    #"lecun_uniform", 18.01 cout
    #"glorot_normal", 18.01 cout
    #"glorot_uniform", 18.01 cout
    #"he_normal", 18.01 cout
    #"lecun_normal", 18.01 cout
    #"he_uniform" 18.01 cout
]

keras_vs_initialisers_config = [
    [2.0, 'fan_in', 'truncated_normal', 'he_normal'],
    [2.0, 'fan_in', 'normal', 'he_normal'],
    [2.0, 'fan_in', 'uniform', 'he_uniform'],
    [1.0, 'fan_in', 'truncated_normal', 'lecun_normal'],
    [1.0, 'fan_in', 'normal', 'lecun_normal'],
    [1.0, 'fan_in', 'uniform', 'lecun_uniform'],
    [1.0, 'fan_avg', 'normal', 'glorot_normal'],
    [1.0, 'fan_avg', 'uniform', 'glorot_uniform']
]
###
# Keras Weight Initialisers
###

keras_losses = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "squared_hinge",
    "hinge",
    "categorical_hinge",
    "logcosh",
    "huber_loss",
    "categorical_crossentropy",
    #"sparse_categorical_crossentropy",
    "binary_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    #"cosine_proximity"
]

###
# Keras Weight Regularisers
###

keras_regularisers = [
    "l1",
    "l2",
    "l1_l2"
]

###
# Operators specific
###

operator_name_dict = {'change_label': 'TCL',
                      'delete_training_data': 'TRD',
                      'unbalance_train_data': 'TUD',
                      'add_noise': 'TAN',
                      'make_output_classes_overlap': 'TCO',
                      'change_batch_size': 'HBS',
                      'change_learning_rate': 'HLR',
                      'change_epochs': 'HNE',
                      'disable_batching': 'HDB',
                      'change_activation_function': 'ACH',
                      'remove_activation_function': 'ARM',
                      'add_activation_function': 'AAL',
                      'add_weights_regularisation': 'RAW',
                      'change_weights_regularisation': 'RCW',
                      'remove_weights_regularisation': 'RRW',
                      'change_dropout_rate': 'RCD',
                      'change_patience': 'RCP',
                      'change_weights_initialisation': 'WCI',
                      'add_bias': 'WAB',
                      'remove_bias': 'WRB',
                      'change_loss_function': 'LCH',
                      'change_optimisation_function': 'OCH',
                      'change_gradient_clip': 'OCG',
                      'remove_validation_set': 'VRM'}


subject_params = {'mnist': {'epochs': 3, 'lower_lr': 0.001, 'upper_lr': 1}, # 11.01 c 12 -> 3
                  'fashion_mnist': {'epochs': 3, 'lower_lr': 0.001, 'upper_lr': 1}, # 17.01 fashion_mnist added juan
                  'cifar': {'epochs': 3, 'lower_lr': 0.001, 'upper_lr': 1}, # 17.01 cifar added
                  'movie_recomm': {'epochs': 5, 'lower_lr': 0.0001, 'upper_lr': 0.001},
                  'audio': {'epochs': 20, 'lower_lr': 0.0001, 'upper_lr': 0.001, 'patience': 10}, # 17.01 50->20 epochs
                  'lenet': {'epochs': 50, 'lower_lr':  0.001, 'upper_lr': 0.01},
                  'udacity': {'epochs': 50, 'lower_lr':  0.00001, 'upper_lr': 0.0001}
                  }

subject_short_name = {'mnist': 'MN', 'fashion_mnist': 'FM', 'cifar':'CF','movie_recomm': 'MR', 'audio': 'SR', 'lenet': 'UE', 'udacity': 'UD'}
