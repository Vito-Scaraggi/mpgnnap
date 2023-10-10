from torch_geometric.loader import DataLoader
import torch
import traceback

from model.mpgnnhandler import MPGNNHandler
from dataset.dataset import CreateDataset
from config.config import cfg

if __name__ == "__main__":
    try:
        print("Reading configs...", end = " ")
        cfg_ = cfg()
        
        torch.manual_seed(cfg_.get_config().seed)
        torch.cuda.manual_seed(cfg_.get_config().seed)
        torch.backends.cudnn.deterministic = True

        print("ok\nLoading dataset...", end = " ")
        dataset = CreateDataset()
        activities_index = dataset.getLabels()
        train_dataset = dataset.getDataset(0)
        validation_dataset = dataset.getDataset(1)
        
        # TODO: includere in package dataset? shuffle a ogni epoca? 
        train_data = DataLoader(train_dataset, batch_size = cfg_.get_config().training.batch_size, shuffle=True)
        validation_data = DataLoader(validation_dataset, batch_size = cfg_.get_config().training.batch_size, shuffle=True)

        print("ok\nCreating model...")
        t = MPGNNHandler(activities_index, cfg_)
        print("Training started")
        t.train(train_data, validation_data)
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())