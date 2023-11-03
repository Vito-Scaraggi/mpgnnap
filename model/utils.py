from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, SortAggregation, Set2Set

class AggrFactory():

    def __init__(self):
        self.aggr_types = {
            'sum': SumAggregation,
            'mean' : MeanAggregation,
            'sort': SortAggregation,
            'set2set': Set2Set,
        }
    
    def factory(self, type, **kwargs):
            """
            Factory method for creating aggregation functions.

            Args:
                type (str): The type of aggregation function to create.
                **kwargs: Additional keyword arguments to pass to the aggregation function.

            Returns:
                An instance of the specified aggregation function.

            Raises:
                ValueError: If an invalid aggregation type is specified or if the specified type
                    does not accept the provided keyword arguments.
            """
            try:
                fun = self.aggr_types[type]
                return fun(**kwargs)
            except KeyError:
                raise ValueError(f"Invalid aggregation type: {type}")
            except TypeError:
                raise ValueError(f"Invalid arguments for aggregation type: {type}")
    
    def dl_in_channels_factor(self, type, **kwargs):
        """
        Computes an adjustment factor for each aggregation type.

        Args:
            type (str): The type of aggregation to be performed. Supported types are "sum", "mean", "sort", and "set2set".
            **kwargs: Additional keyword arguments that may be required for certain aggregation types.

        Returns:
            int: an adjustment factor for the given aggregation type.

        Raises:
            ValueError: If an invalid aggregation type is provided.
        """
        if type == "sum":
            return 1
        elif type == "mean":
            return 1
        elif type == "sort":
            return kwargs["k"]
        elif type == "set2set":
            return 2 
        else:
            raise ValueError(f"Invalid aggregation type: {type}")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Determines whether to stop training early based on the validation loss.

        If the validation loss is lower than the current minimum validation loss, the minimum validation loss is updated
        and the counter is reset to 0. If the validation loss is greater than the minimum validation loss plus a minimum
        delta value, the counter is incremented. If the counter reaches the patience value, training is stopped early.

        Args:
            validation_loss (float): The validation loss to evaluate.

        Returns:
            bool: True if training should be stopped early, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False