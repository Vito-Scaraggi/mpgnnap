'''read file data/meta.pth than plot'''
import torch

def read_train_results():
    file = "data/results_train.pth"
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

def read_test_results():
    file = "data/results_test.pth"
    obj = torch.load(file, map_location=torch.device('cpu'))
    test_results = obj['test_metrics']
    acc = test_results['accuracy']
    prec = test_results['precision']
    rec = test_results['recall']
    f1 = test_results['f1score']
    supp = test_results['support_per_class']

    print(f"Test metrics => Accuracy: {acc*100:.3f}%, Precision: {prec*100:.3f}%, Recall: {rec*100:.3f}%, F1score: {f1*100:.3f}%")
    print(f"Accuracy per class:\n{test_results['accuracy_per_class'].tolist()}")
    print(f"Support per class:\n{supp}")
    
if __name__ == "__main__":
    read_train_results()
    read_test_results()