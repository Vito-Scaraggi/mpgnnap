import traceback
import os

from model.mpgnnhandler import MPGNNHandler
from dataset.dataset import CreateDataset
from config.config import cfg


import sys

if __name__ == "__main__":
    out_dir = "data"
    try:
        team_member = sys.argv[1]
        print("Reading configs...", end = " ")
        cfg_json = cfg().get_config_json()
        
        print("ok\nLoading dataset...", end = " ")
        dataset = CreateDataset()
        activities_index = dataset.getLabels()
        train_dataset = dataset.getDataset(0)
        validation_dataset = dataset.getDataset(1)
        
        if team_member == "chris":
            k = 5
        elif team_member == "luca":
            k = 10
        elif team_member == "vito":
            k = 30
        else:
            raise ValueError("Parametri accettati sono uno tra chris/vito/luca")

        channels = [16, 32, 64]
        n_edge_conv = [2,3,5]
        batch_sizes = [16,32]

        for c in channels:
            for n in n_edge_conv:
                for b in batch_sizes:             
                    cfg_json["model"]["graph_conv_layer_sizes"] = [c] * n
                    cfg_json["model"]["aggregation"]["args"]["k"] = k
                    cfg_json["training"]["batch_size"] = b
                    train_folder = f"k_{str(k)}_c_{str(c)}_n_{str(n)}_b_{str(b)}"
                    outpath = os.path.join(out_dir, train_folder)
                    cfg_ = cfg(cfg_json)
                    t = MPGNNHandler(activities_index, cfg_, outpath)
                    t.train(train_dataset, validation_dataset)
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())