import torch
import numpy as np
import copy
import os.path
from torch_geometric.data import Data, Dataset

PARTIAL_PATH = 'dataset/tensors_files/'

SAVE_DATASET_PATH = 'dataset/data/data_to_load.pt'

class CreateDataset:
    def __init__(self):
        torch.set_printoptions(precision=15)
        pass
    
    def create_tensors(self, phase: int):
        if phase == 0:
            path = 'train.pt'
            file_path = 'training.g'
        elif phase == 1:
            path = 'val.pt'
            file_path = 'val.g'
        elif phase == 2:
            path = 'test.pt'
            file_path = 'test.g'
        else:
            raise ValueError('Unknown value. Allowed values are between 0 and 2.')

        if (not os.path.isfile(PARTIAL_PATH + path[:len(path)-3] + '/e_' + path)) or (not os.path.isfile(PARTIAL_PATH + path[:len(path)-3] + '/n_' + path)) or (not os.path.isfile(PARTIAL_PATH + path[:len(path)-3] + '/gt_' + path)):
            
            labels = self.getLabels()
            
            with open('dataset/' + file_path, 'r') as f:
                contents = f.read()

            graphs = contents.split('\n\n')

            nodes = [] #Array di array di array. Il primo racchiude tutti i grafi; il sotto-array tutti i nodi di quel grafo.
            edges = [] #Stesso di sopra, ma con gli archi.

            # array che contiene tutti i sottografi del grafo in esame
            subgraph_nodes = [] #Array a 4 dimensioni. Quella più esterna racchiude tutti i grafi; quella subito dopo tutti i sottografi
                                #di quel grafo; poi tutti i nodi di quel sottografo.
            subgraph_edges = [] #Stesso di sopra, ma con gli archi.

            #NOTA: le seguenti 3 variabili contengono array di tensori ordinati in modo tale che il tensore i-esimo di ogni
            #variabile fa riferimento allo stesso sottografo.
            nodes_tensor = [] #Array di tensori. Ogni tensore corrisponde a un sottografo; quest'ultimo è caratterizzato dai nodi che
                                #lo compongono, ovvero dalle loro features.
            edges_tensor = [] #Array di tensori. Ogni tensore corrisponde a un sottografo; quest'ultimo è caratterizzato dagli archi che
                                #lo compongono, ovvero dai nodi di partenza e di arrivo.
            ground_truth_tensor = [] #Array di tensori. Ogni tensore contiene i one-hot encoded vector per ogni grafo. 
                                     #L'indice del vettore corrispondente al valore 1 equivale all'indice riferito all'etichetta da predire,
                                     #secondo l'ordine indicato nel file "attributi.txt"

            for i in range(0, len(graphs)):
                graphs[i] = graphs[i][3:] #graphs[i] = grafo i-esimo
                graph_elems = graphs[i].split('\n') #graph_elems = array di stringhe; ogni stringa è un elemento di un grafo

                #Array contenenti rispettivamente i nodi e gli archi del grafo in esame
                graph_nodes = []
                graph_edges = []
                
                #Array contenenti rispettivamente gli indici dei nodi e dei nodi degli archi del grafo in esame
                index_graph_nodes = []
                index_edge_nodes = []

                for j in range(0, len(graph_elems)):
                    if graph_elems[j][0] == 'v':
                        graph_elems[j] = graph_elems[j].split(' ') #graphs_elems[j] = array di stringhe rispettive ai vertici
                                                                    #del grafo i-esimo
                        if '' in graph_elems[j]:
                            graph_elems[j].remove('')
                        graph_elems[j].remove('v')
                        graph_nodes.append(graph_elems[j])
                        index_graph_nodes.append(graph_elems[j][0])

                    if graph_elems[j][0] == 'e':
                        graph_elems[j] = graph_elems[j].split(' ') #graphs_elems[j] = array di stringhe rispettive agli archi
                                                                    #del grafo i-esimo
                        if '' in graph_elems[j]:
                            graph_elems[j].remove('')
                        graph_elems[j].remove('e')
                        graph_edges.append(graph_elems[j])
                        index_edge_nodes.append(graph_elems[j][0])
                        index_edge_nodes.append(graph_elems[j][1])

                #Verifica della correttezza del grafo. Se un arco collega due nodi, di cui almeno uno inesistente,
                #il grafo viene scartato.
                if set(index_edge_nodes).issubset(set(index_graph_nodes)):
                    nodes.append(graph_nodes)
                    edges.append(graph_edges)

                    tmp_nodes = [] #contiene i sottografi del grafo in esame
                    tmp_edges = [] #array di array di array che contiene gli archi dei sottografi del grafo in esame
                    
                    #Generiamo tutti i sottografi del grafo i-esimo, rimuovendo un nodo alla volta, fino ad arrivare al grafo che ha 2 nodi.
                    #Acquisisco i nodi del sottografo k-esimo
                    for k in range(len(graph_nodes)-1, 1, -1):
                        if k == len(graph_nodes)-1:
                            tmp = copy.deepcopy(graph_nodes)
                        else:
                            tmp = copy.deepcopy(tmp)
                        tmp.pop(k)
                        tmp_nodes.append(tmp)
                        
                        tmp_edges2 = [] #array di array che contiene archi di un sottografo
                        
                        #Acquisisco gli archi del sottografo k-esimo
                        for edge in graph_edges:
                            index = len(graph_nodes)-1-k  
                            index_nodes = []
                            
                            #Lista dei nodi del sottografo in esame
                            for node_in_subgraph in tmp_nodes[index]:
                                index_nodes.append(node_in_subgraph[0])
                            
                            #Check se l'arco è contenuto nel sottografo
                            if edge[0] in index_nodes and edge[1] in index_nodes:
                                tmp_edges2.append(edge)
                        
                        tmp_edges.append(tmp_edges2)
                        
                    # array di array di array di array strutturato cosi: 
                    # grafo -> sottografo -> nodi del sottografo -> valori nodo
                    subgraph_nodes.append(tmp_nodes)
                    # array di array di array di array strutturato cosi: 
                    # grafo -> sottografo -> archi del sottografo -> valori arco
                    subgraph_edges.append(tmp_edges) 

            #Creazione del tensore finale a partire dalla lista dei sottografi
            for index, graph in enumerate(subgraph_nodes):
                for  subgraph in graph:
                    subgraph_ground_truth = np.zeros(len(labels)) # quello che sarà vettore di ground truth
                    subgraph_feature = [] #quello che sarà il tensore dei nodi
                    
                    for index2,node in enumerate(subgraph):
                        subgraph_feature.append([float(node[2]), float(node[3]), float(node[4])])
                        
                        #Creazione one-hot encoded vector
                        if index2 == len(subgraph)-1:
                            last_subgraph_node_index = nodes[index].index(node)
                            node_to_predict = nodes[index][last_subgraph_node_index + 1]
                            label_to_predict = node_to_predict[1]
                            index_to_predict = labels.index(label_to_predict)
                            subgraph_ground_truth[index_to_predict] = 1
                
                    #Contiene tutti gli one-hot encoded vectors
                    ground_truth_tensor.append(torch.tensor(subgraph_ground_truth))
                    nodes_tensor.append(torch.tensor(subgraph_feature))

            for graph in subgraph_edges:
                for subgraph in graph:
                    subgraph_e = []
                    start_nodes = []
                    end_nodes = []
                    for edge in subgraph:
                        #-1 perchè l'arco deve far riferimento al nodo e 
                        # in questo caso il numero del nodo è la posizione nell'array 
                        # che quindi inizia da 0 e non da 1
                        start_nodes.append(int(float((edge[0]))) -1) 
                        end_nodes.append(int(float((edge[1]))) -1)
                    subgraph_e.append([start_nodes, end_nodes])

                    #numpy_subgraph_feature = np.array(subgraph_feature)
                    #subgraph_feature_tensor = torch.from_numpy(numpy_subgraph_feature)
                    #nodes_tensor.append(subgraph_feature_tensor)
                    edges_tensor.extend(torch.tensor(subgraph_e))

            os.mkdir(PARTIAL_PATH + path[:len(path)-3])
            torch.save(nodes_tensor, PARTIAL_PATH + path[:len(path)-3] + '/n_' + path)
            torch.save(edges_tensor, PARTIAL_PATH + path[:len(path)-3] + '/e_' + path)
            torch.save(ground_truth_tensor, PARTIAL_PATH + path[:len(path)-3] + '/gt_' + path)

        else:
            nodes_tensor = torch.load(PARTIAL_PATH + path[:len(path)-3] + '/n_' + path)
            edges_tensor = torch.load(PARTIAL_PATH + path[:len(path)-3] + '/e_' + path)
            ground_truth_tensor = torch.load(PARTIAL_PATH + path[:len(path)-3] + '/gt_' + path)

        return nodes_tensor, edges_tensor, ground_truth_tensor
    
    def getLabels(self):
        labels = []
        with open('dataset/attributi.txt', 'r') as file:
                content = file.read()

        labels = content.split('\n')
        return labels

    
    #Transform a graph from tensor to pytorch geometric graph
    def tensor_to_pytorch_graph(self,n,e,gt):
        
        torch_graph = Data(
            x= n,
            edge_index= e,
            y= gt
        )
        return torch_graph
    
    #mode 0: training, 1: validation, 2: test
    def getDataset(self,mode):
        #popola le variabili n, e, gt con i tensori dei grafi
        n, e, gt = self.create_tensors(mode)
        data_list = []
        
        #Creiamo la lista di grafi pytorch geometric
        for i in range (0,len(n)):
            pytorch_graph = self.tensor_to_pytorch_graph(n[i],e[i],gt[i])
            data_list.append(pytorch_graph)
        
        #Convertiamo la lista di data in un dataset
        graphsDataset = GraphDataset(data_list)
        return graphsDataset   

#Dataset personalizzato per caricare i dati che sono sotto forma di List[Data]
class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        super(GraphDataset, self).__init__()

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]