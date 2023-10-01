import torch
import numpy as np
import copy

if __name__ == "__main__":

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

    ground_truth_tensor = []

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

            tmp_nodes = []
            
            #appendiamo i sottografi rimuovendo un nodo alla volta
            for k in range(len(graph_nodes)-1, 1, -1):
                if k == len(graph_nodes)-1:
                    tmp = copy.deepcopy(graph_nodes)
                else:
                    tmp = copy.deepcopy(tmp)
                tmp.pop(k)

                tmp_nodes.append(tmp)
                
            subgraph_nodes.append(tmp_nodes)
            
            #Creare subgraph_edges
            
            #Vanno aggiunti qui invece che sopra ?
            #.extend(subgraph_nodes)
            #.extend(subgraph_edges)
    
    for graph in subgraph_nodes:
        for subgraph in graph:
            subgraph_feature = []
            for node in subgraph:
                subgraph_feature.append([float(node[2]), float(node[3]), float(node[4])])

    

            #numpy_subgraph_feature = np.array(subgraph_feature)
            #subgraph_feature_tensor = torch.from_numpy(numpy_subgraph_feature)
            #nodes_tensor.append(subgraph_feature_tensor)
            nodes_tensor.append(torch.tensor(subgraph_feature))
            #print("TENSOR")
            #print(nodes_tensor)
            
        #break