import traceback
import os
import argparse

from model.mpgnnhandler import MPGNNHandler
from dataset.dataset import CreateDataset
from config.config import cfg

parser = argparse.ArgumentParser(prog='grid search')
#parser.add_argument('team_member', type=str, help='team member name')
parser.add_argument('-s', '--skip', type=int, default=0, help='skip first n combinations')

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        skip = args.skip

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
        
        ks = [5,10,30]
        ch = 32
        n_edge_conv = [2,3,5,7]
        batch_sizes = [16, 32]
        lrs = [0.001, 0.0001]
        

        cfg_json["training"]["learning_rate"] = 0.001
       
        count = 0
        for k in ks:
            for n in n_edge_conv:
                for b in batch_sizes:
                    for lr in lrs:
                        count+=1
                        if count <= skip:
                            continue
                        print(f"\nTraining n° {count}")
                        cfg_json["model"]["graph_conv_layer_sizes"] = [ch] * n
                        cfg_json["model"]["aggregation"]["args"]["k"] = k
                        cfg_json["training"]["learning_rate"] = lr
                        cfg_json["training"]["batch_size"] = b
                        train_folder = f"lr{str(lr)}_b_{str(b)}_n_{str(n)}_k_{str(k)}"
                        outpath = os.path.join(results_path, train_folder)
                        cfg_json["results_path"] = outpath
                        c = cfg(cfg_json)
                        t = MPGNNHandler(activities_index, c)
                        t.train(train_dataset, validation_dataset)
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())