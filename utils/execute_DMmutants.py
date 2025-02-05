import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv
import importlib

import utils.properties as props
import utils.constants as const
import utils.exceptions as e

from utils.mutation_utils import get_accuracy_list_from_scores, update_mutation_properties, load_scores_from_csv
from utils.mutation_utils import concat_params_for_file_name, save_scores_csv, modify_original_model, rename_trained_model
from stats import is_error_rate_within_threshold
from run_deepcrime_properties import read_properties

import numpy as np # for DeepMutation 16.01 juan
import tensorflow as tf # 16.01 juan
from tensorflow.keras.datasets import mnist # for DeepMutation 16.01 juan
from tensorflow.keras.datasets import fashion_mnist # for loadoriginaldata 20.01
from tensorflow.keras.datasets import cifar10 # for loadoriginaldata 20.01


import time # added for computation cost 03.01 juan

scores = []

# load original based on subject 20.01 juan
def loadoriginaldata(dataset_name="mnist"):
    """
    Load test data for different datasets.
    
    Arguments:
    - dataset_name: str, the name of the dataset to load. Options are "mnist", "fashion_mnist", or "cifar10".
    
    Returns:
    - x_test: Test images
    - y_test: Test labels
    """
    if dataset_name == "mnist":
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # normalize
    elif dataset_name == "fashion_mnist":
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # Reshape and normalize
    elif dataset_name == "cifar":
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test, y_test = x_test[:10000], y_test[:10000] # Limit the size of the test set 1000 = 11.4 MB, 10000 = 114 MB
        x_test = x_test.astype('float32') / 255  # Normalize CIFAR-10 data (already in 32x32x3 format)
    else:
        raise ValueError("\nUnsupported dataset name. Choose from 'mnist', 'fashion_mnist', or 'cifar'.\n")
        
    print(f"\nDataset name: {dataset_name} Loaded {x_test.shape[0]} images of shape {x_test.shape[1:]}\n")  # Debug print
    return x_test, y_test

def execute_mutants(mutants_path, mutations):
    global scores
    mutants = []

    for root, dirs, files in os.walk(mutants_path):
        for file in files:
            if file.endswith(".py"):

                mutants.append([root,file])

    for mutation in mutations:

        my_mutants = [mutant for mutant in mutants if mutation in mutant[1]]

        try:
            mutation_params = getattr(props, mutation)
        except AttributeError:
            print("No attributes found")

        model_params = getattr(props, "model_properties")
        udp = [value for key, value in mutation_params.items() if "udp" in key.lower() and "layer" not in key.lower()]

        if len(udp) > 0:
            udp = udp[0]
        else:
            udp = None
        layer_udp = mutation_params.get("layer_udp", None)


        search_type = mutation_params.get("search_type")

        for mutant in my_mutants:
            if mutation_params.get("layer_mutation", False):
                if layer_udp:
                    if isinstance(layer_udp, list):
                        inds = layer_udp
                    else:
                        inds = [layer_udp]
                else:
                    inds = range(model_params["layers_num"])

                for ind in inds:
                #for ind in range(7, 8):
                    # print("index is:" + str(ind))
                    mutation_params["mutation_target"] = None
                    mutation_params["current_index"] = ind
                    mutation_ind = "_" + str(ind)

                    execute_based_on_search(udp, search_type, mutation, mutant, mutation_params, ind, mutation_ind)
            else:
                execute_based_on_search(udp, search_type, mutation, mutant, mutation_params)


def execute_based_on_search(udp, search_type, mutation, mutant, mutation_params, ind = None, mutation_ind = ''):

    global scores
    print(f"\nmutation_params = {mutation_params}\n") # model_name test 17.01 juan
    try:
        if udp and (search_type is None): # 11.01 changed from or -> and 
            original_accuracy_list = get_accuracy_list_from_scores(scores)
            mutation_accuracy_list = get_accuracy_list_from_scores(
                execute_mutant(mutant, mutation_params, mutation_ind))

            is_sts, p_value, effect_size = is_error_rate_within_threshold(mutation_param1, mutation_params, mutation_param2)


            csv_file = os.path.join(mutant[0], "results", "stats", mutation_params['name'] + "_nosearch.csv")

            with open(csv_file, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                if ind:
                    writer.writerow([udp, str(p_value), str(effect_size), str(is_sts)])
                else:
                    writer.writerow([udp, str(p_value), str(effect_size), str(is_sts)])
        elif search_type == 'binary':
            print("calling binary search")
            execute_binary_search(mutant, mutation, mutation_params)
        else:
            print("calling exhaustive search")
            execute_exhaustive_search(mutant, mutation, mutation_params, mutation_ind)

    except (e.AddAFMutationError) as err:
        print(err.message + err.expression)


def execute_mutant(mutant_path, mutation_params, mutation_ind = ''):
    scores = []
    params_list = concat_params_for_file_name(mutation_params)

    # raise  Exception()
    trained_mutants_location = os.path.join(os.getcwd(), const.save_paths["trained"])

    try:
        transformed_path = os.path.join(mutant_path[0], mutant_path[1])
        transformed_path = transformed_path.replace(os.path.sep, ".").replace(".py", "")

        m1 = importlib.import_module(transformed_path)

        data = read_properties()
        if data['mode'] in ('train', 'weak'):
            importlib.reload(m1)

        results_file_path = os.path.join(mutant_path[0], "results", mutant_path[1].replace(".py", "") + "_MP" + params_list + mutation_ind + ".csv")

        if not (os.path.isfile(results_file_path)):
            for i in range(mutation_params["runs_number"]):

                mutation_final_name = mutant_path[1].replace(".py", "") + "_MP" + params_list + mutation_ind + "_" + str(i) + ".h5"

                score = m1.main(mutation_final_name)
                scores.append(score)

                path_trained = [trained_mutants_location,
                                props.model_name + "_trained.h5",
                                mutation_final_name]

                rename_trained_model(path_trained)

            if scores:
                save_scores_csv(scores, results_file_path, params_list)
        else:
            scores = load_scores_from_csv(results_file_path)

    except ImportError as err:
        print('Error:', err)
    else:
        a = 1

    return scores

def execute_original_model(model_path, results_path):
    global scores
    scores = []
    modified_model_path = modify_original_model(model_path)
    
    transformed_path = modified_model_path.replace(os.path.sep, ".").replace(".py", "")

    m1 = importlib.import_module(transformed_path)

    csv_file_path = os.path.join(results_path, props.model_name + ".csv")
    
    if not(os.path.isfile(csv_file_path)):
        for i in range(const.runs_number_default):
        # for i in range(1):
            path_trained = [os.path.join(os.getcwd(), const.save_paths["trained"]),
                            props.model_name + "_trained.h5",
                            props.model_name + "_original_" + str(i) + ".h5"]

            score = m1.main(path_trained[2])

            scores.append(score)

            rename_trained_model(path_trained)
        print("\ntest model path\n")
        
        model_path = (os.path.join(path_trained[0], path_trained[2])) # initialize model path

        print(f"\nmodel_path = {model_path}\n")

        print(f"\nmodel_name = {props.model_name}\n")
        
        # Filter data based on original model's performance
        original_dnn = tf.keras.models.load_model(model_path)

        # Load original test data
        x_test, y_test = loadoriginaldata(props.model_name)

        # If y_test is one-hot encoded, convert it to class indices for comparison
        if y_test.ndim == 2:  # Check if y_test is one-hot encoded (e.g., shape (10000, 10))
            y_test = np.argmax(y_test, axis=1)  # Convert to class indices

        # Get predictions from the original model
        predictions = original_dnn.predict(x_test)

        # Convert predictions to class labels (argmax) and compare with true labels (y_test)
        predicted_labels = predictions.argmax(axis=1)  # classification task

        # Get indices where the model predictions are correct
        correct_indices = np.where(predicted_labels == y_test)[0]

        # Filter the dataset based on correct predictions
        #x_filtered = x_test[correct_indices]
        #y_filtered = y_test[correct_indices]

        # Save the indices of correctly predicted data for use in the mutated model evaluation
        np.save("filtered_indices.npy", correct_indices)  # Save the indices as a file to be used on execute_mutant
        
        save_scores_csv(scores, csv_file_path)
    else:
        print("reading scores from file")
        scores = load_scores_from_csv(csv_file_path)

    return scores


def execute_exhaustive_search(mutant, mutation, my_params, mutation_ind = ''):

    print("Running Exhaustive Search for" + str(mutant))

    original_accuracy_list = get_accuracy_list_from_scores(scores)

    name = my_params['name']
    if name == 'change_optimisation_function':
        for optimiser in const.keras_optimisers:
            print("Changing into optimiser:" + str(optimiser))
            update_mutation_properties(mutation, "optimisation_function_udp", optimiser)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            # mnist_change_optimisation_function_mutated0_MP_sgd_2.h5 # 18.01 juan
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(optimiser), my_params)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(optimiser), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_activation_function' or name == 'add_activation_function':
        for activation in const.activation_functions:
            print("Changing into activation:" + str(activation))
            update_mutation_properties(mutation, "activation_function_udp", activation)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            # mnist_change_activation_function_mutated0_MP_elu_6_0.h5 # 18.01 juan
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(activation), my_params, my_params['current_index'])

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(activation), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_loss_function':
        for loss in const.keras_losses:
            print("Changing into loss:" + str(loss))
            update_mutation_properties(mutation, "loss_function_udp", loss)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(loss), my_params, my_params[''])

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(loss), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_dropout_rate':
        for dropout in const.dropout_values:
            print("Changing into dropout rate:" + str(dropout))
            update_mutation_properties(mutation, "rate", dropout)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(dropout), my_params, my_params[''])

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(dropout), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_batch_size':
        for batch_size in const.batch_sizes:
            print("Changing into batch size:" + str(batch_size))
            update_mutation_properties(mutation, "batch_size", batch_size)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            # mnist_change_batch_size_mutated0_MP_256_256_0.h5    # 18.01 juan
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(batch_size), my_params, str(batch_size))

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(batch_size), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_weights_initialisation':
        for initialiser in const.keras_initialisers:
            print("Changing into initialisation:" + str(initialiser))
            update_mutation_properties(mutation, "weights_initialisation_udp", initialiser)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            # mnist_change_weights_initialisation_mutated0_MP_he_uniform_6_2.h5  # 18.01 juan
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(initialiser), my_params, my_params['current_index'])

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(initialiser), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'add_weights_regularisation':
        for regularisation in const.keras_regularisers:
            print("Changing into regularisation:" + str(regularisation))
            update_mutation_properties(mutation, "weights_regularisation_udp", regularisation)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            
            is_sts, p_value, effect_size = is_error_rate_within_threshold(str(regularisation), my_params, my_params[''])

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(regularisation), str(p_value), str(effect_size), str(is_sts)])


def execute_binary_search(mutant, mutation, my_params):
    print("Running Binary Search for" + str(mutant))

    lower_bound = my_params["bs_lower_bound"]
    upper_bound = my_params["bs_upper_bound"]
    precision = my_params["precision"]
    mutant_name = my_params["name"]

    original_accuracy_list = get_accuracy_list_from_scores(scores)
    update_mutation_properties(mutation, "pct", upper_bound)
    upper_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
    
    if my_params['name'] == 'change_learning_rate': # mnist_change_learning_rate_mutated0_MP_False_0.001_0.h5
        is_sts, p_value, effect_size = is_error_rate_within_threshold('False', my_params, upper_bound)
    else: # mnist_change_epochs_mutated0_MP_3_2.h5 / mnist_change_label_mutated0_MP_100_0.h5
        is_sts, p_value, effect_size = is_error_rate_within_threshold(upper_bound, my_params)

    csv_file = os.path.join(mutant[0], "results", "stats", mutant_name + "_binarysearch.csv")
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow([str(lower_bound), str(upper_bound), '', str(p_value), str(effect_size), str(is_sts)])

    if is_sts:
        print("Binary Search: Upper Bound is Killable")
        search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound, original_accuracy_list)
    else:
        print("Binary Search: Upper Bound is Not Killable")


def search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound, original_accuracy_list):
    if my_params['bs_rounding_type'] == 'int':
        middle_bound = round((upper_bound + lower_bound) / 2)
    elif my_params['bs_rounding_type'] == 'float3':
        middle_bound = round((upper_bound + lower_bound) / 2, 3)
    elif my_params['bs_rounding_type'] == 'float4':
        middle_bound = round((upper_bound + lower_bound) / 2, 4)
    elif my_params['bs_rounding_type'] == 'float5':
        middle_bound = round((upper_bound + lower_bound) / 2, 5)
    else:
        middle_bound = round((upper_bound + lower_bound) / 2, 2)

    print("\nmiddle_bound is:" + str(middle_bound))
    update_mutation_properties(mutation, "pct", middle_bound)
    middle_scores = execute_mutant(mutant, my_params)
    middle_accuracy_list = get_accuracy_list_from_scores(middle_scores)

    is_sts, p_value, effect_size = is_error_rate_within_threshold(middle_bound, my_params)

    csv_file = os.path.join(mutant[0], "results", "stats", my_params["name"] + "_binarysearch.csv")
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow([str(lower_bound), str(upper_bound), str(middle_bound), str(p_value), str(effect_size), str(is_sts)])

    if is_sts:
        upper_bound = middle_bound
    else:
        lower_bound = middle_bound

    if abs(upper_bound - lower_bound) <= my_params['precision']:
        if is_sts:
            perfect = middle_bound
            conf_nk = lower_bound
        else:
            perfect = upper_bound
            conf_nk = middle_bound

        csv_file = os.path.join(mutant[0], "results", "stats", my_params["name"] + "_binarysearch.csv")
        with open(csv_file, 'a') as f1:
            writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
            writer.writerow([str(perfect), str(conf_nk)])

        print("Binary Search Configuration is:" + str(perfect))
        return perfect, conf_nk
    else:
        print("Changing interval to: [" + str(lower_bound) + ", " + str(upper_bound) + "]")
        return search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound,
                                  original_accuracy_list)


