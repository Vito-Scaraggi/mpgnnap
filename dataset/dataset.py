import torch
import numpy as np

if __name__ == "__main__":

    with open('dataset/training.g', 'r') as f:
        contents = f.read()

    graphs = contents.split('\n\n')

    nodes = [] #Array di array di array. Il primo racchiude tutti i grafi; il sotto-array tutti i nodi di quel grafo.
    edges = [] #Stesso di sopra, ma con gli archi.

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
        
        '''
        print("NODES")
        print(nodes)
        print("EDGES")
        print(edges)
        if i==1:
            numpy_nodes = np.array(nodes)
            print("NUMPY_NODES")
            print(numpy_nodes)

            pytorch_tensor = torch.from_numpy(numpy_nodes)
            print("PYTORCH_TENSOR")
            print(pytorch_tensor)
            break
        '''