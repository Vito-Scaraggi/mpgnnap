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
        test_dataset = dataset.getDataset(2)
        test_data = DataLoader(test_dataset, batch_size = cfg_.get_config().test.batch_size, shuffle=True)
        print("ok\nCreating model...")
        t = MPGNNHandler(activities_index, cfg_, True) #add more args maybe
        print("Test started")
        t.evaluate(test_data)
    except Exception as e:
        print(e)
        print(traceback.format_exc())