import torch
import numpy as np
import copy

if __name__ == "__main__":
    
    with open('attributi.txt', 'r') as file:
        content = file.read()
    
    labels = content.split('\n')

    with open('training.g', 'r') as f:
        contents = f.read()

    graphs = contents.split('\n\n')

    nodes = [] #Array di array di array. Il primo racchiude tutti i grafi; il sotto-array tutti i nodi di quel grafo.
    edges = [] #Stesso di sopra, ma con gli archi.

    # array che contiene tutti i sottografi del grafo in esame
    subgraph_nodes = [] #Array a 4 dimensioni. Quella più esterna racchiude tutti i grafi; quella subito dopo tutti i sottografi
                        #di quel grafo; poi tutti i nodi di quel sottografo.
    subgraph_edges = [] #Stesso di sopra, ma con gli archi.

    nodes_tensor = [] #Array di tensori. Ogni tensore corrisponde a un sottografo; quest'ultimo è caratterizzato dai nodi che
                      #lo compongono, ovvero dalle loro features.
    edges_tensor = [] #Array di tensori. Ogni tensore corrisponde a un sottografo; quest'ultimo è caratterizzato dagli archi che
                      #lo compongono, ovvero dai nodi di partenza e di arrivo.

    ground_truth_tensor = [] # contiene i one-hot encoded vector per ogni grafo

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
            
            #appendiamo i sottografi rimuovendo un nodo alla volta
            for k in range(len(graph_nodes)-1, 1, -1):
                if k == len(graph_nodes)-1:
                    tmp = copy.deepcopy(graph_nodes)
                else:
                    tmp = copy.deepcopy(tmp)
                tmp.pop(k)
                tmp_nodes.append(tmp)
                
                tmp_edges2 = [] #array di array che contiene archi di un sottografo
                
                for edge in graph_edges:
                    index = len(graph_nodes)-1-k  
                    index_nodes = []
                    
                    #Lista dei nodi del sottografo in esame
                    for node_in_subgraph in tmp_nodes[index]:
                        index_nodes.append(node_in_subgraph[0])
                    
                    #Check se l'arco è contenuto nel sottografo
                    if edge[0] in index_nodes and edge[1] in index_nodes:
                        tmp_edges2.append(edge)
                
                #appendiamo gli archi del sottografo in esame alla lista di quelli appertenenti al grafo
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
                
                #DA CONTROLLARE se è ok il fatto che possono esserci nodi ripetuti
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
                start_nodes.append(float(edge[0]) -1) 
                end_nodes.append(float(edge[1]) -1)
            subgraph_e.append([start_nodes, end_nodes])

            #numpy_subgraph_feature = np.array(subgraph_feature)
            #subgraph_feature_tensor = torch.from_numpy(numpy_subgraph_feature)
            #nodes_tensor.append(subgraph_feature_tensor)
            edges_tensor.append(torch.tensor(subgraph_e))
    
    
    '''
    with open('nodes.txt', 'w') as f:
        f.write(str(nodes_tensor))
    
    with open('edges.txt', 'w') as f:
        f.write(str(edges_tensor))
    
    with open('ground_truth.txt', 'w') as f:
        f.write(str(ground_truth_tensor))
    '''

    