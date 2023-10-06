import torch
from model.simplempgnn import SimpleMPGNN

class MPGNNTrainer():

    def __init__(self, activities_index, dropout_rate = 0.1):
        num_node_features = 3
        graph_conv_layer_sizes = [32, 32, 32]
        dense_layer_sizes = [32, 32]
        
        aggr_type = "sort" # "sum", "set2set"
        aggr_args = {"k": 3} # {} , {"in_channels" : 32, "processing_steps" : 5 }

        self.model = SimpleMPGNN(num_node_features, 
                            graph_conv_layer_sizes,
                            dense_layer_sizes, 
                            dropout_rate, 
                            activities_index,
                            aggr_type,
                            aggr_args
                            )

    def train(self, train_data, validation_data, batch_size, epochs, learning_rate = 0.001):

        self.train_data = train_data
        self.validation_data = validation_data

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.L1Loss()
    
        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []

        epoch_losses = []
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            num_correct_train = 0
            total_train = 0
            
            for b in self.train_data: # Trainings
                batch = b.to(self.model.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.model(batch)
                label = batch.y.view(out.shape[0],-1) # out.shape[0] to get the size of the current batch. (Last batch can be smaller if batch_size does not divide the number of instances)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                #calculate train accuracy
                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                total_train += len(predictions)
                num_correct_train += torch.sum(predictions == ground_truth).item()

            
            train_accuracy = num_correct_train/total_train
            self.train_accuracies.append(train_accuracy)

            valid_loss = 0

            self.model.eval()
            
            num_correct_validation = 0
            total_validation = 0

            with torch.no_grad():
                for b in self.validation_data: # Validation
                    batch = b.to(self.model.device)
                    out = self.model(batch)
                    label = batch.y.view(out.shape[0],-1)
                    loss = criterion(out, label)
                    valid_loss += loss.item()

                    #calculate validation accuracy
                    predictions = torch.argmax(out, dim=1)
                    ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                    total_validation += len(predictions)
                    num_correct_validation += torch.sum(predictions == ground_truth).item()
            
            validation_accuracy = num_correct_validation/total_validation
            self.validation_accuracies.append(validation_accuracy)
            
            this_epoch_losses = (train_loss/len(self.train_data), valid_loss/len(self.validation_data)) # Average loss over the batches
            epoch_losses.append(this_epoch_losses)

            self.train_losses.append(this_epoch_losses[0])
            self.validation_losses.append(this_epoch_losses[1])

            print(f"Epoch {epoch+1} completed. Avg train. Loss: {this_epoch_losses[0]}, Avg valid. Loss: {this_epoch_losses[1]};", end = " ")
            print(f"Train. Accuracy: {train_accuracy*100:.4f}%, Valid. Accuracy: {validation_accuracy*100:.4f}%")

    def evaluate(self, data) -> float:
        
        num_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for b in data:
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                total += len(predictions)
                num_correct += torch.sum(predictions == ground_truth).item()

        accuracy = num_correct/total
        print(f"Test Accuracy: {accuracy*100:.4f}%")

    #aggiungere salvataggio pesi model best