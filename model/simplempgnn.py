import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Module
from torch_geometric.nn import MessagePassing, EdgeConv
from torch_geometric.utils import sort_edge_index
from typing import List

from model.utils import AggrFactory


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
            num_node_features: int,
            graph_conv_layer_sizes: List[int],
            dense_layer_sizes: List[int],
            dropout_rate: float,
            #learning_rate: float,
            activities_index: List[str],
            aggr_type : str = "sum",
            aggr_args : dict = None,
            use_cuda_if_available: bool = True,
        ):

        super().__init__()

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda:0") #TODO: Add support for multiple GPUs?
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.graph_conv_layer_sizes = graph_conv_layer_sizes
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        #self.learning_rate = learning_rate

        self.num_features = num_node_features
        # number of nodes ?
        self.num_output_features = len(activities_index) # One-hot encoding of the activity

        # Graph Convolutions
        self.conv1 = EdgeConv(self.num_features, graph_conv_layer_sizes[0])
        self.convs = torch.nn.ModuleList()
        for in_size, out_size in zip(graph_conv_layer_sizes, graph_conv_layer_sizes[1:]):
            self.convs.append(EdgeConv(in_size, out_size))
        
        aggr_factory = AggrFactory()
        self.aggr_layer  = aggr_factory.factory(aggr_type, **aggr_args)

        # Dense Layers
        # Input size from the source code
        #TODO: If len(dense_layer_sizes) == 0 this would fail
        self.linear = torch.nn.Linear( aggr_factory.dl_in_channels_factor(aggr_type, **aggr_args) * graph_conv_layer_sizes[-1], dense_layer_sizes[0])
        self.linears = torch.nn.ModuleList()
        for in_size, out_size in zip(dense_layer_sizes, dense_layer_sizes[1:]):
            self.linears.append(Linear(in_size, out_size))
        self.linear_output = Linear(dense_layer_sizes[-1], self.num_output_features) # Final layer for output

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        
        if hasattr(self.aggr_layer, "reset_parameters"):
            self.aggr_layer.reset_parameters()
        
        self.linear.reset_parameters()
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = sort_edge_index(edge_index, sort_by_row = False)
        # Graph Convolution
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        x = self.aggr_layer(x)
        
        # Reshape the tensor to be able to pass it to the dense layers (flatten ?)
        x = torch.flatten(x)
        
        # Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

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
            "Dense Layer Sizes": self.dense_layer_sizes,
            "Dropout Rate": self.dropout_rate,
            #"Learning Rate": self.learning_rate,
            "Distinct Activities": self.num_output_features,
        }
        return self.__class__.__name__ + " Model: " + str(params)