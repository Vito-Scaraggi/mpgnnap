from torch_geometric.loader import DataLoader

from model.mpgnntrainer import MPGNNTrainer
from dataset.dataset import CreateDataset

if __name__ == "__main__":

    # read params from config file?
    batch_size = 8
    epochs = 3
    learning_rate = 0.01
    dropout_rate = 0.1
    
    print("Loading dataset...")
    dataset = CreateDataset()
    activities_index = dataset.getLabels()
    train_dataset = dataset.getDataset(0)
    validation_dataset = dataset.getDataset(1)
    test_dataset = dataset.getDataset(2)
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    t = MPGNNTrainer(activities_index, dropout_rate) #add more args maybe
    print("Training started")
    t.train(train_data, validation_data, batch_size, epochs, learning_rate)
    print("Testing")
    t.evaluate(test_data)