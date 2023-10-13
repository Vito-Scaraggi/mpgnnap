import traceback
import os
import argparse

from model.mpgnnhandler import MPGNNHandler
from dataset.dataset import CreateDataset
from config.config import cfg

parser = argparse.ArgumentParser(prog='gridsearch')
parser.add_argument('team_member', type=str, help='team member name')
parser.add_argument('-s', type=int, default=0, help='skip first n combinations')

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        team_member = args.team_member
        skip = args.s

        print("Reading configs...", end = " ")
        c = cfg()
        cfg_ = c.get_config()
        cfg_json = c.get_config_json()
        
        results_path = cfg_.results_path

        print("ok\nLoading dataset...", end = " ")
        dataset = CreateDataset(cfg_.dataset_path)
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
        #k = [5,10,30]
        
       
        count = 0
        for c in channels:
            for n in n_edge_conv:
                for b in batch_sizes:
                    count+=1
                    if count <= skip:
                        continue
                    cfg_json["model"]["graph_conv_layer_sizes"] = [c] * n
                    cfg_json["model"]["aggregation"]["args"]["k"] = k
                    cfg_json["training"]["batch_size"] = b
                    train_folder = f"k_{str(k)}_c_{str(c)}_n_{str(n)}_b_{str(b)}"
                    c = cfg(cfg_json)
                    outpath = os.path.join(results_path, train_folder)
                    t = MPGNNHandler(activities_index, c)
                    t.train(train_dataset, validation_dataset)
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())