import torch
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='plot results', description='Plot results from training and testing')
parser.add_argument("-p","--path", type=str, default="results/exp", help='path to results folder')

def read_train_results(path):
    """
    Reads the training results from the specified path and returns the train and validation losses.

    Args:
        path (str): The path to the directory containing the training results.

    Returns:
        tuple: A tuple containing the train and validation losses.
    """
    file = os.path.join(path, "results_train.pth")
    obj =  torch.load(file, map_location=torch.device('cpu'))
    
    train_losses = obj['train_losses']
    validation_losses = obj['validation_losses']
    
    #train_results = obj['train_metrics']
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
    """
    Reads the test results from the specified path and returns the accuracy per class.

    Args:
        path (str): The path to the directory containing the test results.

    Returns:
        list: A list of floats representing the accuracy per class.
    """
    file = os.path.join(path, "results_test.pth")
    obj = torch.load(file, map_location=torch.device('cpu'))
    
    test_results = obj['test_metrics']
    acc = test_results['accuracy']
    prec = test_results['precision']
    rec = test_results['recall']
    f1 = test_results['f1score']
    conf_matrix = test_results['confusion_matrix_per_class']
    print(f"Confusion matrix:\n{conf_matrix.tolist()}")

    acc_per_class = test_results['accuracy_per_class'].tolist()
    f1_per_class = test_results['f1score_per_class'].tolist()

    print(f"Test metrics => Accuracy: {acc*100:.3f}%, Precision: {prec*100:.3f}%, Recall: {rec*100:.3f}%, F1score: {f1*100:.3f}%")
    return acc_per_class, f1_per_class
    
if __name__ == "__main__":
    args = parser.parse_args()
    results_path = args.path
    train_losses, validation_losses = read_train_results(results_path)
    accuracy_per_class, f1score_per_class = read_test_results(results_path)
    
    '''plot train and validation loss over epochs'''

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
    plt.ylabel("Accuracy %")
    plt.xticks(range(24))
    plt.title("Accuracy per class in test")
    
    for i, v in enumerate(accuracy_per_class):
        plt.text(i-0.25, v + 0.025, f"{(v*100):.2f}", color='black', fontweight='bold', size = 'small')
    plt.show()

    '''plot barchart of F1score per class with 24 classes.
    Each bar is labeled with value of F1score per class and the class number'''
    plt.figure()
    plt.bar(range(24), f1score_per_class)
    plt.xlabel("Class")
    plt.ylabel("F1score %")
    plt.xticks(range(24))
    plt.title("F1score per class in test")
    
    for i, v in enumerate(f1score_per_class):
        plt.text(i-0.25, v + 0.025, f"{(v*100):.2f}", color='black', fontweight='bold', size = 'small')
    plt.show()