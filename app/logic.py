import os
import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml

from app.algo import Coordinator, Client


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.models = {}
        self.splits = {}
        self.test_splits = {}
        self.betas = {}
        self.train = None
        self.test = None
        self.pred_output = None
        self.test_output = None
        self.label_column = None
        self.sep = None
        self.split_mode = None
        self.split_dir = None

    def read_config(self):
        try:
            with open(self.INPUT_DIR + '/config.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)['fc_linear_regression']

                self.train = config['input']['train']
                self.test = config['input']['test']
                self.pred_output = config['output']['pred']
                self.test_output = config['output']['test']
                self.label_column = config['format']['label']
                self.sep = config['format']['sep']
                self.split_mode = config['split']['mode']
                self.split_dir = config['split']['dir']

            if self.split_mode == "directory":
                self.splits = dict.fromkeys(
                    [f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.split_dir}') if f.is_dir()])
                self.test_splits = dict.fromkeys(self.splits.keys())
                self.models = dict.fromkeys(self.splits.keys())
                self.betas = dict.fromkeys(self.splits.keys())
            else:
                self.splits[self.INPUT_DIR] = None

            for split in self.splits.keys():
                os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)

            shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
            print(f'Read config file.', flush=True)
        except:
            print(f'N config file found. Please use the frontend to select input.', flush=True)

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_local_computation = 3
        state_wait_for_aggregation = 4
        state_global_aggregation = 5
        state_writing_results = 6
        state_finishing = 7

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("[CLIENT] Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", self.coordinator)

            if state == state_read_input:
                self.progress = "read input"
                print('[CLIENT] Read input and config')
                self.read_config()

                for split in self.splits.keys():
                    if self.coordinator:
                        self.models[split] = Coordinator()
                    else:
                        self.models[split] = Client()
                    train_path = split + "/" + self.train
                    test_path = split + "/" + self.test
                    X = pd.read_csv(train_path, sep=self.sep)
                    y = X.loc[:, self.label_column]
                    X = X.drop(self.label_column, axis=1)

                    X_test = pd.read_csv(test_path, sep=self.sep)
                    y_test = X_test.loc[:, self.label_column]
                    X_test = X_test.drop(self.label_column, axis=1)

                    y_test.to_csv(split.replace("/input/", "/output/") + "/" + self.test_output, index=False)

                    self.splits[split] = [X, y]
                    self.test_splits[split] = [X_test, y_test]

                    state = state_local_computation
            if state == state_local_computation:
                print("[CLIENT] Perform local computation")
                self.progress = 'local computation'
                data_to_send = {}
                for split in self.splits.keys():
                    model = self.models[split]

                    xtx, xty = model.local_computation(self.splits[split][0], self.splits[split][1])

                    data_to_send[split] = [xtx, xty]
                data_to_send = jsonpickle.encode(data_to_send)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_aggregation:
                print("[CLIENT] Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregation data from coordinator.")
                    global_coefs = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []
                    for split in self.splits:
                        self.models[split].set_coefs(global_coefs[split])

                    state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("[CLIENT] Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    for split in self.splits:
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        aggregated_beta = self.models[split].aggregate_beta(split_data)
                        self.models[split].set_coefs(aggregated_beta)
                        self.betas[split] = aggregated_beta

                    data_to_broadcast = jsonpickle.encode(self.betas)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting computation data to clients', flush=True)

            if state == state_writing_results:
                print("[CLIENT] Writing results")
                # now you can save it to a file
                for split in self.splits:
                    model = self.models[split]
                    joblib.dump(model, split.replace("/input", "/output") + '/model.pkl')
                    y_pred = pd.DataFrame(model.predict(self.test_splits[split][0]), columns=["y_pred"])
                    y_pred.to_csv(split.replace("/input", "/output") + "/" + self.pred_output, index=False)

                if self.coordinator:
                    self.data_incoming.append('DONE')
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("Finishing")
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            time.sleep(1)


logic = AppLogic()
