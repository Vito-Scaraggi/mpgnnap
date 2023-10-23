import torch
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='read & plot')
parser.add_argument("-p","--path", type=str, default="results/exp", help='path to results folder')

def read_train_results(path):
    file = os.path.join(path, "results_train.pth")
    obj =  torch.load(file, map_location=torch.device('cpu'))
    
    train_losses = obj['train_losses']
    validation_losses = obj['validation_losses']
    
    train_results = obj['train_metrics']
    validation_results = obj['validation_metrics']

    acc = validation_results['accuracies']
    prec = validation_results['precisions']
    rec = validation_results['recalls']
    f1 = validation_results['f1scores']
    
    max_index = torch.argmax(torch.tensor(acc)).item()
    print(f"Best epoch: {max_index+1}")
    print(f"Best val metrics => Accuracy: {acc[max_index]*100:.3f}%, Precision: {prec[max_index]*100:.3f}%, Recall: {rec[max_index]*100:.3f}%, F1score: {f1[max_index]*100:.3f}%")

    return train_losses, validation_losses

def read_test_results(path):
    file = os.path.join(path, "results_test.pth")
    obj = torch.load(file, map_location=torch.device('cpu'))
    test_results = obj['test_metrics']
    acc = test_results['accuracy']
    prec = test_results['precision']
    rec = test_results['recall']
    f1 = test_results['f1score']
    #supp = test_results['support_per_class']

    print(f"Test metrics => Accuracy: {acc*100:.3f}%, Precision: {prec*100:.3f}%, Recall: {rec*100:.3f}%, F1score: {f1*100:.3f}%")
    #print(f"Accuracy per class:\n{test_results['accuracy_per_class'].tolist()}")
    
    return test_results['accuracy_per_class'].tolist()
    
if __name__ == "__main__":
    args = parser.parse_args()
    results_path = args.path
    train_losses, validation_losses = read_train_results(results_path)
    accuracy_per_class = read_test_results(results_path)
    
    plt.figure()
    plt.plot(train_losses, label = "train")
    plt.plot(validation_losses, label = "validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Average train and validation loss over batches")
    plt.show()

    '''plot barchart of accuracy per class with 24 classes.
    Each bar is labeled with value of accuracy per class and the class number'''
    plt.figure()
    plt.bar(range(24), accuracy_per_class)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.xticks(range(24))
    plt.title("Accuracy per class in test")
    for i, v in enumerate(accuracy_per_class):
        plt.text(i-0.25, v + 0.025, f"{(v*100):.2f}", color='black', fontweight='bold', size = 'small')
    plt.show()