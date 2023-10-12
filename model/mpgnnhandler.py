import torch
import os
import json

from model.simplempgnn import SimpleMPGNN
from model.metrics import Metrics
from model.utils import EarlyStopper

class MPGNNHandler():

    def __init__(self, activities_index, cfg, load_weights = False):
        try:
            self.cfg = cfg.get_config()
            self.model = SimpleMPGNN(activities_index, self.cfg.model)
            self.model = self.model.to(self.model.device)
            
            self.save_model_path = "data/model_best.pth"
            self.save_meta_path = "data/meta.pth"
            self.save_results_path = f"data/results_{'test' if load_weights else 'train'}.pth"

            if not os.path.exists("data"):
                os.makedirs("data")
            
            if load_weights:
                self.meta = torch.load(self.save_meta_path)
                print("Loaded model: " + json.dumps(self.meta, indent=4))
                self.model.load_state_dict(torch.load(self.save_model_path, map_location=self.model.device))
            else:
                print("Configs: " + json.dumps(cfg.get_config_json(), indent=4))
                self.meta =  cfg.get_config_json()
        
        except FileNotFoundError:
            raise FileNotFoundError("No weights found")        

    def train(self, train_data, validation_data):
        train_cfg = self.cfg.training

        self.train_data = train_data
        self.validation_data = validation_data

        self.batch_size = train_cfg.batch_size
        self.epochs = train_cfg.epochs
        self.learning_rate = train_cfg.learning_rate

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.L1Loss()
    
        self.train_losses = []
        self.validation_losses = []

        self.train_metrics = Metrics(self.model.num_output_features, self.model.device)
        self.validation_metrics = Metrics(self.model.num_output_features, self.model.device)

        epoch_losses = []

        model_best = None
        best_metric = 0.0
        early_stopper = EarlyStopper(patience= train_cfg.early_stop.patience, min_delta=train_cfg.early_stop.min_delta)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for b in self.train_data: # Trainings
                batch = b.to(self.model.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.model(batch)
                label = batch.y.view(out.shape[0],-1) # out.shape[0] to get the size of the current batch. (Last batch can be smaller if batch_size does not divide the number of instances)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                gt = torch.argmax(label, dim=1).to(self.model.device)
                self.train_metrics.update(pred, gt)
            
            valid_loss = 0

            self.model.eval()
            
            with torch.no_grad():
                for b in self.validation_data: # Validation
                    batch = b.to(self.model.device)
                    out = self.model(batch)
                    label = batch.y.view(out.shape[0],-1)
                    loss = criterion(out, label)
                    valid_loss += loss.item()
                    pred = torch.argmax(out, dim=1)
                    gt = torch.argmax(label, dim=1).to(self.model.device)
                    self.validation_metrics.update(pred, gt)
            
            #save epoch losses
            this_epoch_losses = (train_loss/len(self.train_data), valid_loss/len(self.validation_data)) # Average loss over the batches
            epoch_losses.append(this_epoch_losses)
            self.train_losses.append(this_epoch_losses[0])
            self.validation_losses.append(this_epoch_losses[1])

            #save epoch metrics
            self.train_metrics.save_epoch_metrics()
            self.validation_metrics.save_epoch_metrics()

            new_val_metric = self.validation_metrics.get_main_metric()
            
            if new_val_metric > best_metric:
                print(f"[New best]", end = " ")
                best_metric = new_val_metric
                model_best = self.model.state_dict()


            print(f"Epoch {epoch+1} completed. LR: {optimizer.state_dict()['param_groups'][-1]['lr']}, Avg train. Loss: {this_epoch_losses[0]}, Avg valid. Loss: {this_epoch_losses[1]}")
            print("Train metrics:", end = " ")
            self.train_metrics.print_metrics()
            print("Valid metrics:", end = " ")
            self.validation_metrics.print_metrics()

            self.train_metrics.reset()
            self.validation_metrics.reset()
            if early_stopper.early_stop(this_epoch_losses[1]):             
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        # saving model best to file
        torch.save(model_best, self.save_model_path)
        
        # saving meta to file
        torch.save(self.meta, self.save_meta_path)

        self.results = {}
        self.results["train_losses"] = self.train_losses
        self.results["validation_losses"] = self.validation_losses
        self.results["train_metrics"] = self.train_metrics.get_state()
        self.results["validation_metrics"] = self.validation_metrics.get_state()
        
        # save results to file 
        torch.save(self.results, self.save_results_path)

    def evaluate(self, data) -> float:
        
        self.test_metrics = Metrics(self.model.num_output_features, self.model.device)
        self.model.eval()

        with torch.no_grad():
            for b in data:
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape)
                pred = torch.argmax(out, dim=1)
                gt = torch.argmax(label, dim=1).to(self.model.device)
                self.test_metrics.update(pred, gt)
                
            self.test_metrics.save_epoch_metrics()
            self.test_metrics.print_metrics()
            
            self.results = {}
            self.results["test_metrics"] = self.test_metrics.get_state(True) 
            torch.save(self.results, self.save_results_path)