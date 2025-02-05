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
from mutation_score import calculate_dc_ms, calculate_dpp_ms, calculate_dm_ms   # added calculate_dpp_ms, calculate_dm_ms 

data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate():

    data['subject_name'] = 'mnist'
    data['subject_path'] = os.path.join('test_models', 'mnist_conv.py')
    
    #data['mutations'] = ["change_label", "delete_training_data", "unbalance_train_data", "add_noise",  "change_epochs", "change_learning_rate", "change_activation_function", "change_weights_initialisation", "change_optimisation_function", "change_batch_size",]  # 10 DC:ok 
    
    #data['mutations'] = ["change_label"]  #B DC:ok DPP:ok DM:ok 11.12
    #data['mutations'] = ["delete_training_data"]  #B DC:ok DPP:ok DM:ok 11.12
    
    #data['mutations'] = ["unbalance_train_data"] #B DC:unbalance_train_data"ok DPP:ok DM:ok 11.12 (https://github.com/dlfaults/deepcrime/issues/2)
    #data['mutations'] = ["add_noise"] #B DC:ok DPP:ok DM:ok 11.12 -> DC: x 03.01
    #data['mutations'] = ["make_output_classes_overlap"] #B DC:ok DPP:ok DM:ok 11.12
    
    data['mutations'] = ["change_batch_size"] #EU ([Errno 2] No such file or directory: 'mutated_models/mnist/results_train/stats/change_batch_size_exssearch.csv') -> found solution same1 (o with and) DC:ok DPP:ok DM:x 11.12 

    #data['mutations'] = ["change_epochs"] #B DC:ok DPP:ok DM:ok 11.12
    #data['mutations'] = ["change_learning_rate"] #B DC:ok DPP:ok DM:ok 11.12
    #data['mutations'] = ["disable_batching"] #- x (super().__init__(activity_regularizer=activity_regularizer, **kwargs) Epoch 1/2 Killed) -> find solution same2 (o with and) # DC:ok DPP:ok DM:ok 11.12
    #data['mutations'] = ["change_activation_function"] #EL x ([Errno 2] No such file or directory: 'mutated_models/mnist/results_train/stats/change_activation_function_exssearch.csv') -> find solution same3 (o with and) DC:ok DPP:ok DM:x 11.12

    #data['mutations'] = ["remove_activation_function"] #EL ok DC:ok DPP:ok DM:ok 12.12
    #data['mutations'] = ["change_weights_initialisation"] #EL o (with extra settings in operators 27.11) ok DC:ok DPP:ok DM:x 12.12
    #data['mutations'] = ["change_optimisation_function"] #EL o (adadelta comment out why?? 1h) DC:ok DPP:ok DM:ok 12.12
    #data['mutations'] = ["remove_validation_set"] #- DC:ok DPP:ok DM:ok 12.12
    #data['mutations'] = ["remove_bias"] #EL o? (check paper) DC:ok, no res DPP:ok, no res DM:ok, no res 12.12
    
    
###############################################################################################################
    
    #data['mutations'] = ["change_loss_function"] # DC:x DPP:x DM:x (TypeError: string indices must be integers, not 'str') 12.12
    #data['mutations'] = ["change_dropout_rate"] # 0? DC:ok, no res DPP:ok, no res DM:ok, no res 11.12
    #data['mutations'] = ["add_weights_regularisation"] # DC:x DPP:x DM:x (Exception encountered: Could not interpret regularizer identifier: l1_l2) 12.12
    
    #data['mutations'] = ["change_earlystopping_patience"] # NA to mnist
    #data['mutations'] = ["add_activation_function"] # NA to mnist
    #data['mutations'] = ["add_bias"] # NA to any
    #data['mutations'] = ["change_gradient_clip"] # NA to any
    #data['mutations'] = ["change_weights_regularisation"] # NA to any
    #data['mutations'] = ["remove_weights_regularisation"] # NA to any
    dc_props.write_properties(data)
    '''
    shutil.copyfile(os.path.join('utils', 'properties', 'properties_example.py'),
                    os.path.join('utils', 'properties.py'))
    shutil.copyfile(os.path.join('utils', 'properties', 'constants_example.py'),
                    os.path.join('utils', 'constants.py'))

    importlib.reload(props)
    importlib.reload(const)
    '''
    run_deepcrime_tool()

    if props.MS == 'DC_MS':
        data['mode'] = 'train'
        data['subject_path'] = os.path.join('test_models', 'mnist_conv_train.py')
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


        run_deepcrime_tool()

        if os.path.isdir(test_results):
            if os.path.isdir(test_results + '_train'):
                shutil.rmtree(test_results + '_train')
            shutil.move(test_results, test_results + '_train')
        else:
            raise Exception()
        # test ms 15.01.2025
        train_accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_train")
        accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_test")
        
        calculate_dc_ms(train_accuracy_dir, accuracy_dir)
        #calculate_dpp_ms(train_accuracy_dir, accuracy_dir)
        #calculate_dm_ms(train_accuracy_dir, accuracy_dir)
    print("Finished all, exit")


if __name__ == '__main__':
    run_automate()