from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class Metrics():
    def __init__(self, num_classes, device):
        self.metrics = {        
            "accuracy" : MulticlassAccuracy(num_classes = num_classes, average = "micro" ).to(device),
            "accuracy_per_class" : MulticlassAccuracy(num_classes = num_classes, average = None).to(device),
            "precision" : MulticlassPrecision(num_classes = num_classes, average = "macro").to(device),
            "precision_per_class" : MulticlassPrecision(num_classes = num_classes, average = None).to(device),
            "recall" : MulticlassRecall(num_classes = num_classes, average = "macro" ).to(device),
            "recalls_per_class" : MulticlassRecall(num_classes = num_classes, average = None).to(device),
            "f1score" : MulticlassF1Score(num_classes = num_classes, average = "macro" ).to(device),
            "f1score_per_class" : MulticlassF1Score(num_classes = num_classes, average = None).to(device),
        }

        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1scores = []

    def update(self, pred, gt):
        for metric in self.metrics.values():
            metric.update(pred, gt)

    def save_epoch_metrics(self):
        self.accuracies.append(self.metrics['accuracy'].compute())
        self.precisions.append(self.metrics['precision'].compute())
        self.recalls.append(self.metrics['recall'].compute())
        self.f1scores.append(self.metrics['f1score'].compute())

    # main metric is F1score
    def get_main_metric(self):
        # return self.accuracy.compute()
        return self.metrics["f1score"].compute()

    def print_metrics(self):
        for key, val in self.metrics.items():
            if not key.__contains__("per_class"):
                strvalue = f"{val.compute()*100:.3f}".rjust(7)
                print(f"{key.capitalize()}: {strvalue}%", end = ", ")
        print()
    
    def get_state(self, complete = False):
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
        for metric in self.metrics.values():
            metric.reset()