#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os # for DeepMutation 15.01 juan
import glob
import csv
import re
import tensorflow as tf
from tensorflow.keras.datasets import mnist # for DeepMutation 29.11 juan
from tensorflow.keras.datasets import fashion_mnist # for loadoriginaldata 20.01
from tensorflow.keras.datasets import cifar10 # for loadoriginaldata 20.01

from run_deepcrime_properties import read_properties # for model_name 17.01

import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt

from patsy import dmatrices
import statsmodels.stats.power as pw
from scipy.stats import wilcoxon
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from utils.exceptions import InvalidStatisticalTest
from utils import properties

import time # for computational cost 17.01 juan

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
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # Reshape and normalize
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

#calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(orig_accuracy_list, ddof=1) ** 2 + (ny-1)*np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result

#calculates whether two accuracy arrays are statistically different according to GLM (=> DeepCrime criteria 15.01. juan)
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold = 0.05):
    # Calculate execution time only once
    if not hasattr(is_diff_sts, "execution_time"):
        start_time = time.time()
    
    print(f"\nOriginal accuracy list: {orig_accuracy_list}\n") # test 15.01
    print(f"mutant accuracy list: {accuracy_list}\n") # test 15.01
    
    if properties.statistical_test == "WLX":
        p_value = p_value_wilcoxon(orig_accuracy_list, accuracy_list)
    elif properties.statistical_test == "GLM":
        p_value = p_value_glm(orig_accuracy_list, accuracy_list)
    else:
        raise InvalidStatisticalTest("The selected statistical test is invalid/not implemented.")

    effect_size = cohen_d(orig_accuracy_list, accuracy_list)

    if properties.model_type == 'regression':
        is_sts = ((p_value < threshold) and effect_size <= -0.5)
    else:
        is_sts = ((p_value < threshold) and effect_size >= 0.5)

    # Calculate execution time only once
    if not hasattr(is_diff_sts, "execution_time"):
        end_time = time.time()
        is_diff_sts.execution_time = end_time - start_time
        print(f"\nDC_execution_time = {is_diff_sts.execution_time}\n")
    
    return is_sts, p_value, effect_size

# DM++(DPP) mutant selection criteria 15.01 juan
def is_accuracy_threshold(orig_accuracy_list, accuracy_list):
    """
    Applies DeepMutation++ mutant selection criteria.
    The mutant is included if its accuracy is at least 90% of the original model's accuracy on the filtered (correctly classified) test data.
    Returns:
        - inclusion: True if the mutant is included, False otherwise.
        - p_value: Dummy value (0) as statistical test is not performed.
        - effect_size: Dummy value (0) as effect size is not computed.
    """
    # Calculate execution time only once
    if not hasattr(is_accuracy_threshold, "execution_time"):
        start_time = time.time()
    
    dc_props = read_properties()
    Ofile_path = dc_props['subject_path']
    model_name = dc_props['subject_name']
    
    #print(f"\nfile_path = {Ofile_path}\n") # file_path = test_models/fashion_mnist_conv.py   17.01 juan
    #print(f"\nmodel_name = {model_name}\n") # model_name = fashion_mnist                    17.01 juan
    
    # Load the indices of the correctly classified data from the original model (as used in DeepMutation)
    correct_indices = np.load("filtered_indices.npy")
    (x_test, y_test) = loadoriginaldata(model_name)

    # Filter the test data based on the correct predictions from the original model
    x_filtered = x_test[correct_indices]
    y_filtered = y_test[correct_indices]
    
    # Set the file path for the original model
    file_path = f"/home/jovyan/BT_test/deepcrime2/trained_models/{model_name}_original_0.h5"  # Path to the original model file
    
    #print(f"\nfile_path = {file_path}\n")  # file_path test
    
    # Load the original model (ensure correct path is used)
    original_model = tf.keras.models.load_model(file_path)  # Load the model from the passed path
    original_predictions = original_model.predict(x_filtered)
    original_accuracy = np.mean(np.argmax(original_predictions, axis=1) == y_filtered)  # Original accuracy on the filtered data
    
    accuracy_threshold_multiplier=0.9
    print(f"\nOriginal model accuracy on filtered data = {original_accuracy}\n")
    print(f"\nAccuracy Threshold = {original_accuracy*accuracy_threshold_multiplier}\n")
    # Mean accuracy of the mutant on the filtered data
    mutant_accuracy = np.mean(np.array(accuracy_list))  # Mutant's accuracy (already calculated)
    
    print(f"\nMutant accuracy on filtered data = {mutant_accuracy}\n")


    # DeepMutation++ filter: mutant must have accuracy >= 90% of the original model's accuracy on the filtered data
    if mutant_accuracy >= original_accuracy * accuracy_threshold_multiplier:
        print("\nMutant included\n")
        inclusion = True
    else:
        print("\nMutant excluded\n")
        inclusion = False
        
    # Calculate execution time only once
    if not hasattr(is_accuracy_threshold, "execution_time"):
        end_time = time.time()
        is_accuracy_threshold.execution_time = end_time - start_time
        print(f"\nDPP_execution_time = {is_accuracy_threshold.execution_time}\n")
        
    return inclusion, 0, 0  # Inclusion-exclusion

# DM mutant selection criteria 15.01 juan
def is_error_rate_within_threshold(mutation_param1, mutation_params, mutation_param2="none"):
    """
    Determines whether the mutant model's error rate is below the specified threshold.
    The mutant is included if its error rate is less than or equal to the threshold.

    :param mutant: The mutated model (path or model object).
    :param mutation_params: Mutation parameters (for potential future expansion).
    :param threshold: The error rate threshold (e.g., 0.2 for 20%).
    :return: Boolean indicating whether the mutant is included based on error rate and the error rate value.
    """
    # Calculate execution time only once
    if not hasattr(is_error_rate_within_threshold, "execution_time"):
        start_time = time.time()

    dc_props = read_properties()
    Ofile_path = dc_props['subject_path']
    model_name = dc_props['subject_name']
    
    #print(f"\nfile_path = {Ofile_path}\n") # file_path = test_models/fashion_mnist_conv.py   17.01 juan
    #print(f"\nmodel_name = {model_name}\n") # model_name = fashion_mnist                    17.01 juan
    
    # Load the indices of the correctly classified data
    correct_indices = np.load("filtered_indices.npy")
    x_test, y_test = loadoriginaldata(model_name)  # Function to load data

    # Filter the test data based on correct predictions
    x_filtered = x_test[correct_indices]
    y_filtered = y_test[correct_indices]
    
    mutation_file_path = ""
    if mutation_param2 == "none":
        mutation_file_path = f"/home/jovyan/BT_test/deepcrime2/trained_models/{model_name}_{mutation_params['name']}_mutated0_MP_{mutation_param1}_2.h5"
    else:
        mutation_file_path = f"/home/jovyan/BT_test/deepcrime2/trained_models/{model_name}_{mutation_params['name']}_mutated0_MP_{mutation_param1}_{mutation_param2}_2.h5"  # Path to the mutation model file

    # Load the mutated model
    model = tf.keras.models.load_model(mutation_file_path)
    
    # Get predictions from the mutated model on the filtered test data
    mutant_predictions = model.predict(x_filtered)

    # Calculate the error rate for the mutant
    incorrect_predictions = np.where(mutant_predictions.argmax(axis=1) != y_filtered)[0]
    error_rate_mutant = len(incorrect_predictions) / len(x_filtered)

    print(f"\nError rate for the mutant {mutation_file_path}: {error_rate_mutant}\n")
    
    threshold=0.2
    # Include the mutant if its error rate is below the threshold
    if error_rate_mutant <= threshold:
        print("\nMutant included\n")
        inclusion = True
    else:
        print("\nError rate exceeds threshold. Mutant excluded.\n")
        inclusion = False

    # Calculate execution time only once
    if not hasattr(is_error_rate_within_threshold, "execution_time"):
        end_time = time.time()
        is_error_rate_within_threshold.execution_time = end_time - start_time
        print(f"\nDM_execution_time = {is_error_rate_within_threshold.execution_time}\n")
        
    return inclusion, 0, 0


def p_value_wilcoxon(orig_accuracy_list, accuracy_list):
    w, p_value_w = wilcoxon(orig_accuracy_list, accuracy_list)

    return p_value_w


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(effect_size=eff_size, nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow
