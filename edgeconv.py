import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Module, Conv1d
from torch_geometric.nn import MessagePassing, knn_graph, EdgeConv
from typing import List

# implementation of "Dynamic Graph CNN for Learning on Point Clouds" paper in pytorch geometric
class EdgeConv(MessagePassing):
    # N nodes
    # E edges
    # in_channels = features
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        
        # building multi-layer perceptron
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    # build all messages to node i
    def message(self, x_i, x_j):
        # i is the node that aggregates information
        # j is its neighbour
        
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels] 

        # concatenates tensors
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

class SimpleMPGNN(Module):

    def __init__(self,
            #dataset: Dataset,
            num_node_features: int,
            graph_conv_layer_sizes: List[int],
            sizes_1d_convolutions: List[int],
            dense_layer_sizes: List[int],
            dropout_rate: float,
            learning_rate: float,
            activities_index: List[str],
            use_cuda_if_available: bool = True
        ):

        """A Deep Graph Convolutional Neural Network (DGCNN) for next-activity prediction. Based on the paper "Exploiting Instance Graphs and Graph Neural Networks for Next Activity Prediction" by Chiorrini et al.

        Args:
            dataset (Dataset): The dataset to construct the model for.
            graph_conv_layer_sizes (List[int]): The output size for each graph convolution layer.
            sort_pool_k (int): The number of nodes to select in the SortPooling layer.
            sizes_1d_convolutions (List[int]): The output size for each 1D convolution layer.
            dense_layer_sizes (List[int]): The output size for each dense layer.
            dropout_rate (float): The dropout rate to use in the Dropout layer after the 1D convolution.
            learning_rate (float): The learning rate to use for the optimizer.
            activities_index (List[str]): The list of activities in the log. The same as used for one-hot encoding. Used to determine the number of input features.
            use_cuda_if_available (bool, optional): Whether to use CUDA if available. Defaults to True. If CUDA is not available, CPU will be used either way.
        """


        super().__init__()

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda:0") #TODO: Add support for multiple GPUs?
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.graph_conv_layer_sizes = graph_conv_layer_sizes
        self.sizes_1d_convolutions = sizes_1d_convolutions
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        #TODO: Research kernel size in a bit more detail...
        
        # KERNEL_SIZE è la dimensione del kernel nelle conv 1d = numero di graph conv layer?
        #self.KERNEL_SIZE = len(graph_conv_layer_sizes)

        self.num_features = num_node_features
        # number of nodes ?
        self.num_output_features = len(activities_index) # One-hot encoding of the activity

        # Graph Convolutions
        self.conv1 = EdgeConv(self.num_features, graph_conv_layer_sizes[0])
        self.convs = torch.nn.ModuleList()
        for in_size, out_size in zip(graph_conv_layer_sizes, graph_conv_layer_sizes[1:]):
            self.convs.append(EdgeConv(in_size, out_size))

        '''
        # In forward, there is some tensor magic between these two "layers"

        # 1D Convolution
        self.conv1d = Conv1d(graph_conv_layer_sizes[-1], sizes_1d_convolutions[0], self.KERNEL_SIZE)
        self.conv1ds = torch.nn.ModuleList()
        for in_size, out_size in zip(sizes_1d_convolutions, sizes_1d_convolutions[1:]):
            self.conv1ds.append(Conv1d(in_size, out_size, self.KERNEL_SIZE))

        # Dropout done in `forward`
        '''

        # Dense Layers
        # Input size from the source code
        #TODO: If len(dense_layer_sizes) == 0 this would fail
        self.linear = torch.nn.Linear( self.num_output_features * graph_conv_layer_sizes[-1], dense_layer_sizes[0])
        self.linears = torch.nn.ModuleList()
        for in_size, out_size in zip(dense_layer_sizes, dense_layer_sizes[1:]):
            self.linears.append(Linear(in_size, out_size))
        self.linear_output = Linear(dense_layer_sizes[-1], self.num_output_features) # Final layer for output

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        for conv1d in self.conv1ds:
            conv1d.reset_parameters()
        self.linear.reset_parameters()
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, x, edge_index):
        # Graph Convolution
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        print(x.shape)

        '''
        # Weird tensor magic from the source code
        x = x.view(len(x), self.num_output_features, -1).permute(0, 2, 1) # modification of the structure of the vector to be able to pass it to the conv1d layer (they must have n°nodes=k) (translated from their source code)

        # qui x deve avere dimensione (n°nodes, n° canali, n°features)
        # 1D Convolutions
        x = F.relu(self.conv1d(x))
        for conv1d in self.conv1ds:
            x = F.relu(conv1d(x))
        '''

        # Reshape the tensor to be able to pass it to the dense layers (flatten ?)
        #x = x.view(len(x), -1)
        x = torch.flatten(x)
        print(x.shape)

        # Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        #print(x.shape)
        # Dense Layers
        x = F.relu(self.linear(x))
        for linear in self.linears:  
            x = F.relu(linear(x))
        x = F.relu(self.linear_output(x))
        x = torch.softmax(x, dim=0) # Softmax to get probabilities
        return x # No activation function, as pytorch already applies softmax in cross-entropy loss

    def __repr__(self):
        params = {
            "Graph Conv. Layer Sizes": self.graph_conv_layer_sizes,
            "1D Conv. Sizes": self.sizes_1d_convolutions,
            "Dense Layer Sizes": self.dense_layer_sizes,
            "Dropout Rate": self.dropout_rate,
            "Learning Rate": self.learning_rate,
            "Distinct Activities": self.num_output_features,
        }
        return self.__class__.__name__ + " Model: " + str(params)

    def train(self):
        '''load data and train the model'''
        pass
    
    def _training_loop(self):
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

if __name__ == "__main__":
    num_node_features = 2
    graph_conv_layer_sizes = [32, 64, 64]
    size_1d_convolutions = [32, 16]
    dense_layer_sizes = [32]
    dropout_rate = 0.1
    learning_rate = 0.01
    activities_index = ["A", "B", "C"]
    model = SimpleMPGNN(num_node_features, graph_conv_layer_sizes, size_1d_convolutions, dense_layer_sizes, dropout_rate, learning_rate, activities_index)

    x = torch.tensor([[1.,0.], [0., 1.], [.5, .5]], dtype=torch.float32)
    # edge_index from to, gather and scatter
    edge_index = torch.tensor([[0, 1],[1, 2]], dtype=torch.int64) # c3 graph
    y = model(x, edge_index)
    print(y)