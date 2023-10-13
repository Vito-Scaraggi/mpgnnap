import traceback

from model.mpgnnhandler import MPGNNHandler
from dataset.dataset import CreateDataset
from config.config import cfg

if __name__ == "__main__":
    try:
        print("Reading configs...", end = " ")
        cfg_ = cfg()

        print("ok\nLoading dataset...", end = " ")
        dataset = CreateDataset()
        activities_index = dataset.getLabels()
        train_dataset = dataset.getDataset(0)
        validation_dataset = dataset.getDataset(1)

        print("ok\nCreating model...")
        t = MPGNNHandler(activities_index, cfg_)
        print("Training started")
        t.train(train_dataset, validation_dataset)
    
    except Exception as e:
        print(e)
        print(traceback.format_exc())