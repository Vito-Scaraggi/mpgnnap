from typing import Any
import torch
from torch_geometric.loader import DataLoader
from model.simplempgnn import SimpleMPGNN

class MPGNNTrainer():

    def __init__(self, activities_index, dropout_rate = 0.1):
        num_node_features = 3
        graph_conv_layer_sizes = [32, 32, 32]
        dense_layer_sizes = [32, 32]
        
        aggr_type = "sort" # "set2set"
        aggr_args = {"k": 3} # {"in_channels" : 32, "processing_steps" : 5 }

        self.model = SimpleMPGNN(num_node_features, 
                            graph_conv_layer_sizes,
                            dense_layer_sizes, 
                            dropout_rate, 
                            activities_index,
                            aggr_type,
                            aggr_args
                            )

    def train(self, train_dataset, validation_dataset, batch_size, epochs, learning_rate = 0.001):
        
        self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_data = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model.train(True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.L1Loss()

        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []

        epoch_losses = []
        for epoch in range(self.epochs):
            self.model.train(True)
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

            valid_loss = 0

            for b in self.validation_data: # Validation
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape[0],-1)
                loss = criterion(out, label)
                valid_loss += loss.item()
            
            this_epoch_losses = (train_loss/len(self.train_data), valid_loss/len(self.validation_data)) # Average loss over the batches
            epoch_losses.append(this_epoch_losses)

            valid_accuracy = self.evaluate(self.validation_data)
            print(f"Epoch {epoch+1} completed. Train. Loss: {this_epoch_losses[0]}, Valid. Loss: {this_epoch_losses[1]}; Valid. Accuracy: {valid_accuracy*100:.4f}%")

            self.train_losses.append(this_epoch_losses[0])
            self.validation_losses.append(this_epoch_losses[1])

            self.train_accuracies.append(self.evaluate(self.train_data)) # model set to eval in here. No problem since we set train at the top
            self.validation_accuracies.append(valid_accuracy)


    def evaluate(self, test_dataset, batch_size) -> float:
        
        num_correct = 0
        total = 0
        self.test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.model.eval()

        with torch.no_grad():
            for b in self.test_data:
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                total += len(predictions)
                num_correct += torch.sum(predictions == ground_truth).item()

        self.test_accuracy = num_correct/total
        print(f"Accuracy on Test Set: {self.test_accuracy:.4f}%")