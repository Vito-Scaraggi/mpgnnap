from model.mpgnntrainer import MPGNNTrainer

if __name__ == "__main__":
    batch_size = 16
    epochs = 10
    learning_rate = 0.001
    dropout_rate = 0.1
    # set activities_index, train_dataset, validation_dataset, test_dataset

    t = MPGNNTrainer(activities_index, dropout_rate) #add more args maybe
    t.train(train_dataset, validation_dataset, batch_size, epochs, learning_rate)
    t.evaluate(test_dataset, batch_size)