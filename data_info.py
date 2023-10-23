import os
import torch
import matplotlib.pyplot as plt

from config.config import cfg

def get_supports(m):
    c = cfg()
    cfg_ = c.get_config()
    
    dataset_path = cfg_.dataset_path
    num_activities = 24
    
    supports = { i : 0 for i in range(num_activities) }
    
    path_gt = os.path.join(dataset_path, m, f"gt_{m}.pt")
    
    data_gt = torch.load(path_gt)

    for label in data_gt:
        supports[torch.argmax(label).item()] += 1
    
    supports_perc = { i : round(supports[i]/len(data_gt),4) for i in range(num_activities) }
    print(f"{len(data_gt)} samples.\n")
    #print(f"Supports per class:\n{supports}\n")
    #print(f"Supports % per class\n{supports_perc}\n")

    return supports, supports_perc

def ig_stats(m):
    file = m + ".g"
    if m == "train":
        file = "training.g"

    with open('_data/' + file, 'r') as f:
            contents = f.read()

    graphs = contents.split('\n\n')

    print(f"Number of instance graphs: {len(graphs)}")

    num_nodes_per_graph = [0 for _ in range(len(graphs))]

    for n, graph in enumerate(graphs):
        graph = graph[3:]
        graph_elems = graph.split('\n') #graph_elems = array di stringhe; ogni stringa è un elemento di un grafo
        
        for elem in graph_elems:
            if elem[0] == 'v':
                num_nodes_per_graph[n] += 1
    
    print(f"Mean number of nodes per ig: {sum(num_nodes_per_graph)/len(num_nodes_per_graph):.2f}")
    print(f"Standard deviation of number of nodes per ig: {torch.std(torch.tensor(num_nodes_per_graph, dtype = torch.float64)):.2f}")
    print(f"Min number of nodes per ig: {min(num_nodes_per_graph)}")
    print(f"Max number of nodes per ig: {max(num_nodes_per_graph)}")


if __name__ == '__main__':
    modes = ["train","val", "test"]
    plot_supports = True

    for m in modes:
        print(f"{m.capitalize()} dataset")
        ig_stats(m)
        print("After preprocessing:", end = " ")
        supports, supports_perc = get_supports(m)

        if plot_supports:
            """
                plot barchart of supports percentages per class with 24 classes.
                Each bar is labeled with value of support per class and the class number
                Set the title with the mode name
            """
            plt.figure()
            plt.bar(range(24), supports_perc.values())
            plt.xlabel("Class")
            plt.ylabel("Support %")
            plt.xticks(range(24))
            for i, v in enumerate(supports_perc.values()):
                plt.text(i-0.25, v + 0.005, f"{(v*100):.2f}", color='black', fontweight='bold', size = 'small')
            plt.title(f"Support percentage per class in {m} dataset")
            plt.show()

            plt.figure()
            plt.bar(range(24), supports.values())
            plt.xlabel("Class")
            plt.ylabel("Support")
            plt.xticks(range(24))
            for i, v in enumerate(supports.values()):
                plt.text(i-0.25, v + 10, f"{v}", color='black', fontweight='bold', size = 'small')
            plt.title(f"Support per class in {m} dataset")
            plt.show()