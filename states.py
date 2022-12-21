import os
import shutil
import threading
import time

import numpy as np
import joblib
import jsonpickle
import pandas as pd
import yaml

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State
from algo import Coordinator, Client

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log("[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'
        
        
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('local computation', Role.BOTH)
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Read input and config")
            self.read_config()
        
            splits = self.load('splits')
            test_splits = self.load('test_splits')
            models = self.load('models')
        
            for split in splits.keys():
                if self.is_coordinator:
                    models[split] = Coordinator()
                else:
                    models[split] = Client()
                train_path = split + "/" + self.load('train')
                test_path = split + "/" + self.load('test')
                X = pd.read_csv(train_path, sep=self.load('sep'))
                y = X.loc[:, self.load('label_column')]
                X = X.drop(self.load('label_column'), axis=1)

                X_test = pd.read_csv(test_path, sep=self.load('sep'))
                y_test = X_test.loc[:, self.load('label_column')]
                X_test = X_test.drop(self.load('label_column'), axis=1)

                y_test.to_csv(split.replace("/input", "/output") + "/" + self.load('test_output'), index=False)

                splits[split] = [X, y]
                test_splits[split] = [X_test, y_test]
        
            self.store('iteration', 0)
            return 'local computation'
            
        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return 'read input'
        
    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        models = {}
        splits = {}
        test_splits = {}
        betas = {}
        
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_linear_regression']

            self.store('train', config['input']['train'])
            self.store('test', config['input']['test'])
            self.store('pred_output', config['output']['pred'])
            self.store('test_output', config['output']['test'])
            self.store('label_column', config['format']['label'])
            self.store('sep', config['format']['sep'])
            self.store('split_mode', config['split']['mode'])
            self.store('split_dir', config['split']['dir'])
            self.store('smpc_used', config.get('use_smpc', False))

        if self.load('split_mode') == "directory":
            splits = dict.fromkeys(
                    [f.path for f in os.scandir(f"{self.load('INPUT_DIR')}/{self.load('split_dir')}") if f.is_dir()])
            test_splits = dict.fromkeys(splits.keys())
            models = dict.fromkeys(splits.keys())
            betas = dict.fromkeys(splits.keys())
        else:
            splits[self.load('INPUT_DIR')] = None

        for split in splits.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)

        shutil.copyfile(self.load('INPUT_DIR') + '/config.yml', self.load('OUTPUT_DIR') + '/config.yml')
        self.log(f'Read config file.')
            
        self.store('models', models)
        self.store('splits', splits)
        self.store('test_splits', test_splits)
        self.store('betas', betas)
            
        
@app_state('local computation', Role.BOTH)
class LocalComputationState(AppState):
    """
    Perform local computation and send the computation data to the coordinator.
    """
    
    def register(self):
        self.register_transition('global aggregation', Role.COORDINATOR)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        self.register_transition('local computation', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Perform local computation")
            data_to_send = []
            splits = self.load('splits')
            models = self.load('models')

            for split in splits.keys():
                model = models[split]
                xtx, xty = model.local_computation(splits[split][0], splits[split][1])
                if self.load('smpc_used'):
                    data_to_send.append([self.to_list(xtx), self.to_list(xty)])
                else:
                    data_to_send.append([xtx, xty])

            if self.load('smpc_used'):
                self.configure_smpc()
            self.send_data_to_coordinator(data_to_send, use_smpc=self.load('smpc_used'))

            if self.is_coordinator:
                return 'global aggregation'
            else:
                self.log(f'[CLIENT] Sending computation data to coordinator')
                return 'wait for aggregation'
 
        except Exception as e:
            self.log('error local computation', LogLevel.ERROR)
            self.update(message='error local computation', state=State.ERROR)
            print(e)
            return 'local computation'
 
    def to_list(self, np_array):
        if not isinstance(np_array, (np.ndarray, list)):
            if isinstance(np_array, np.float64):
                return np_array.item()
            return np_array
        lst_arr = []
        for item in np_array:
            if isinstance(np_array, (np.ndarray, list)):
                non_np = self.to_list(item)
            else:
                non_np = item
            lst_arr.append(non_np)
        return lst_arr

    
@app_state('wait for aggregation', Role.PARTICIPANT)
class WaitForAggregationState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """
    
    def register(self):
        self.register_transition('writing results', Role.PARTICIPANT)
        self.register_transition('wait for aggregation', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Wait for aggregation")
            global_coefs = self.await_data()
            self.log("[CLIENT] Received aggregation data from coordinator.")
            models = self.load('models')
            for idx, split in enumerate(self.load('splits')):
                models[split].set_coefs(global_coefs[idx])

            return 'writing results'
        
        except Exception as e:
            self.log('error wait for aggregation', LogLevel.ERROR)
            self.update(message='error wait for aggregation', state=State.ERROR)
            print(e)
            return 'wait for aggregation'

# GLOBAL PART

@app_state('global aggregation', Role.COORDINATOR)
class GlobalAggregationState(AppState):
    """
    The coordinator receives the local computation data from each client and aggregates the weights.
    The coordinator broadcasts the global computation data to the clients.
    """
    
    def register(self):
        self.register_transition('writing results', Role.COORDINATOR)
        self.register_transition('global aggregation', Role.COORDINATOR)
        
    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Global computation")
            if self.load('smpc_used'):
                data = self.aggregate_data(use_smpc=True)
            else:
                data = self.gather_data()
            models = self.load('models')
            betas = self.load('betas')
            data_to_broadcast = []
            for idx, split in enumerate(self.load('splits')):
                if self.load('smpc_used'):
                    aggregated_beta = models[split].aggregate_beta(self.to_numpy(data[idx][0]), self.to_numpy(data[idx][1]), True)
                else:
                    split_data = []
                    for client in data:
                        split_data.append(client[idx])
                    XT_X = [client[0] for client in split_data]
                    XT_y = [client[1] for client in split_data]
                    aggregated_beta = models[split].aggregate_beta(XT_X, XT_y, False)
                models[split].set_coefs(aggregated_beta)
                betas[split] = aggregated_beta
                data_to_broadcast.append(aggregated_beta)
            self.broadcast_data(data_to_broadcast, send_to_self=False)
            self.log(f'[CLIENT] Broadcasting computation data to clients')

            return 'writing results'
        
        except Exception as e:
            self.log('error global aggregation', LogLevel.ERROR)
            self.update(message='error global aggregation', state=State.ERROR)
            print(e)
            return 'global aggregation'
    
    def to_numpy(self, lst):
        if isinstance(lst, list):
            np_arr = []
            for item in lst:
                np_item = item
                if isinstance(item, list):
                    np_item = self.to_numpy(item)
                np_arr.append(np_item)
            try:
                return np.array(np_arr, dtype='float64')
            except:
                return np.array(np_arr, dtype='object')
        return lst


@app_state('writing results', Role.BOTH)
class WritingResultsState(AppState):
    """
    Writes the results of the global computation.
    """
    
    def register(self):
        self.register_transition('terminal', Role.BOTH)
        self.register_transition('writing results', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            self.log("[CLIENT] Writing results")
            # now you can save it to a file
            models = self.load('models')
            test_splits = self.load('test_splits')
            for split in self.load('splits'):
                model = models[split]
                joblib.dump(model, split.replace("/input", "/output") + '/model.pkl')
                y_pred = pd.DataFrame(model.predict(test_splits[split][0]), columns=["y_pred"])
                y_pred.to_csv(split.replace("/input", "/output") + "/" + self.load('pred_output'), index=False)

            self.send_data_to_coordinator('DONE')
        
            if self.is_coordinator:
                self.log("Finishing")
                self.gather_data()
            
            return 'terminal'
        
        except Exception as e:
            self.log('error writing results', LogLevel.ERROR)
            self.update(message='error writing results', state=State.ERROR)
            print(e)
            return 'writing results'
