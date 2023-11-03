from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchmetrics.classification import MulticlassConfusionMatrix

class Metrics():

    def __init__(self, num_classes, device):
        """
        Initializes a Metrics object with various metrics for evaluating model performance.

        Args:
            num_classes (int): The number of classes in the classification task.
            device (torch.device): The device on which to perform computations.
        """
        self.metrics = {        
            "accuracy" : MulticlassAccuracy(num_classes = num_classes, average = "micro" ).to(device),
            "accuracy_per_class" : MulticlassAccuracy(num_classes = num_classes, average = None).to(device),
            "precision" : MulticlassPrecision(num_classes = num_classes, average = "macro").to(device),
            "precision_per_class" : MulticlassPrecision(num_classes = num_classes, average = None).to(device),
            "recall" : MulticlassRecall(num_classes = num_classes, average = "macro" ).to(device),
            "recalls_per_class" : MulticlassRecall(num_classes = num_classes, average = None).to(device),
            "f1score" : MulticlassF1Score(num_classes = num_classes, average = "macro" ).to(device),
            "f1score_per_class" : MulticlassF1Score(num_classes = num_classes, average = None).to(device),
            "confusion_matrix_per_class" : MulticlassConfusionMatrix(num_classes = num_classes).to(device),
        }

        self.num_classes = num_classes        
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1scores = []

    def update(self, pred, gt):
        """
        Updates the metrics with the given predictions and ground truth values.

        Args:
            pred: The predicted values.
            gt: The ground truth values.
        """
        for metric in self.metrics.values():
            metric.update(pred, gt)

    def save_epoch_metrics(self):
        """
        Saves the accuracy, precision, recall, and F1 score for the current epoch.

        The metrics are computed using the `compute()` method of the corresponding metric objects
        and appended to the `accuracies`, `precisions`, `recalls`, and `f1scores` lists, respectively.
        """
        self.accuracies.append(self.metrics['accuracy'].compute())
        self.precisions.append(self.metrics['precision'].compute())
        self.recalls.append(self.metrics['recall'].compute())
        self.f1scores.append(self.metrics['f1score'].compute())

    def get_main_metric(self):
        """
        Returns the computed accuracy metric for the model.
        """
        return self.metrics["accuracy"].compute()
    
    def print_metrics(self):
        """
        Prints the computed metrics for the model.

        The method iterates over the metrics dictionary and prints the computed metrics for each metric that does not
        contain the string "per_class" in its key.

        Returns:
            None
        """
        for key, val in self.metrics.items():
            if not key.__contains__("per_class"):
                strvalue = f"{val.compute()*100:.3f}".rjust(7)
                print(f"{key.capitalize()}: {strvalue}%", end = ", ")
        print()
    
    def get_state(self, complete=False):
        """
        Returns a dictionary containing the current state of the metrics object.

        Args:
            complete (bool): If True, computes and includes the values of all metrics.

        Returns:
            dict: A dictionary containing the current state of the metrics object.
                  If `complete` is True, the dictionary will include the computed values of all metrics.
                  Otherwise, the dictionary will only include the values of the metrics per epoch.
        """
        state = {}
        if complete:
            for key, val in self.metrics.items():
                state[key] = val.compute()

        state["accuracies"] = self.accuracies
        state["precisions"] = self.precisions
        state["recalls"] = self.recalls
        state["f1scores"] = self.f1scores

        return state

    def reset(self):
        """
        Resets all metrics to their initial state.
        """
        for metric in self.metrics.values():
            metric.reset()