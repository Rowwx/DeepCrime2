import os
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib

import utils.properties as props
import utils.constants as const
import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool
from utils.constants import save_paths
from mutation_score import calculate_dc_ms, calculate_dpp_ms, calculate_dm_ms, calculate_ms   # added calculate_dpp_ms, calculate_dm_ms  also calculate_ms 18.01 juan
import time # for computation time 18.01 juan
data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate(subject_name, subject_path, subject_path_train, msc): # msc = 'DC' or 'DPP' or 'DM' 18.01 juan

    data['subject_name'] = subject_name
    data['subject_path'] = os.path.join('test_models', subject_path)
    
    data['mutations'] = ["change_label", "delete_training_data", "unbalance_train_data", "add_noise", "change_batch_size", "change_epochs", "change_learning_rate", "change_activation_function", "change_weights_initialisation", "change_optimisation_function"]  # 10 
    # MN -> DC:ok (11.01), DPP: ok (19.01), DM: ok (19.01)
    # FM -> DC:ok (19.01), DPP: ok (19.01), DM: ok (19.01)
    # CF -> DC:ok (19.01), DPP: ok (19.01), DM: ok (19.01)
    
    #data['mutations'] = ["change_label"]  #B DC:ok (11.01) DPP: x (18.01) DM:ok 
    #data['mutations'] = ["delete_training_data"]  #B DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["unbalance_train_data"] #B DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["add_noise"] #B DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["change_batch_size"] #EU DC:ok (11.01) DPP:ok DM:x  
    #data['mutations'] = ["change_epochs"] #B DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["change_learning_rate"] #B DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["change_activation_function"] #EL DC:ok (11.01) DPP:ok DM:ok 
    #data['mutations'] = ["change_weights_initialisation"] #EL DC:ok (11.01) DPP:ok DM:x 
    #data['mutations'] = ["change_optimisation_function"] #EL DC:ok (11.01) DPP:ok DM:ok 
    
###############################################################################################################
    
    #data['mutations'] = ["disable_batching"] #- x (super().__init__(activity_regularizer=activity_regularizer, **kwargs) Epoch 1/2 Killed) -> find solution same2 (o with and) # DC:ok DPP:ok DM:ok 11.12
    #data['mutations'] = ["make_output_classes_overlap"] #B DC no Mutation Score 17.01
    #data['mutations'] = ["add_weights_regularisation"] # DC:x DPP:x DM:x (Exception encountered: Could not interpret regularizer identifier: l1_l2) 12.12
    #data['mutations'] = ["change_dropout_rate"] #EL 0? DC:ok, no res DPP:ok, no res DM:ok, no res 11.12
    #data['mutations'] = ["remove_validation_set"] #- 
    #data['mutations'] = ["change_loss_function"] # DC: X 17.01
    
    #data['mutations'] = ["remove_activation_function"] #EL (not implemented)
    #data['mutations'] = ["remove_bias"] #EL (not implemented)
    #data['mutations'] = ["change_earlystopping_patience"] # NA to mnist
    #data['mutations'] = ["add_activation_function"] # NA to mnist
    #data['mutations'] = ["add_bias"] # NA to any
    #data['mutations'] = ["change_gradient_clip"] # NA to any
    #data['mutations'] = ["change_weights_regularisation"] # NA to any
    #data['mutations'] = ["remove_weights_regularisation"] # NA to any
    
    dc_props.write_properties(data)

    run_deepcrime_tool(msc)

    if props.MS == 'DC_MS':
        data['mode'] = 'train'
        data['subject_path'] = os.path.join('test_models', subject_path_train)
        dc_props.write_properties(data)
        
        # test ms 15.01.2025
        test_results = os.path.join(data['root'], save_paths['mutated'],  data['subject_name'], 'results')
        print(test_results)

        if os.path.isdir(test_results):
            if os.path.isdir(test_results + '_test'):
                shutil.rmtree(test_results + '_test')
            shutil.move(test_results, test_results + '_test')
        else:
            raise Exception()

        start_time = time.time() # start time
        
        run_deepcrime_tool(msc)
        
        end_time = time.time() # end time
        comp_time = end_time - start_time
        print(f"\n {subject_name} Computation_time = {comp_time}\n")

        if os.path.isdir(test_results):
            if os.path.isdir(test_results + '_train'):
                shutil.rmtree(test_results + '_train')
            shutil.move(test_results, test_results + '_train')
        else:
            raise Exception()
        # test ms 15.01.2025
        train_accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_train")
        accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_test")

        calculate_ms(train_accuracy_dir, accuracy_dir, comp_time, msc)
        
        #calculate_dc_ms(train_accuracy_dir, accuracy_dir)
        #calculate_dpp_ms(train_accuracy_dir, accuracy_dir)
        #calculate_dm_ms(train_accuracy_dir, accuracy_dir)
    print("Finished all, exit")


if __name__ == '__main__':
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DPP')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DM')
    # run 2
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DC')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DPP')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DM')
    
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DC')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DPP')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DM')
    
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DC')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DPP')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DM')
    # run 3
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DC')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DPP')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DM')
    
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DC')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DPP')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DM')
    
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DC')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DPP')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DM')
    # run 4
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DC')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DPP')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DM')
    
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DC')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DPP')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DM')
    
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DC')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DPP')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DM')
    # run 5
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DC')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DPP')
    run_automate('mnist', 'mnist_conv.py', 'mnist_conv_train.py', 'DM')
    
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DC')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DPP')
    run_automate('fashion_mnist', 'fashion_mnist_conv.py', 'fashion_mnist_conv_train.py', 'DM')
    
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DC')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DPP')
    run_automate('cifar', 'cifar_conv.py', 'cifar_conv_train.py', 'DM')
    
    