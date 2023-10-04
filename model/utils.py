from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, SortAggregation, Set2Set, AttentionalAggregation

class AggrFactory():

    def __init__(self):
        self.aggr_types = {
            'sum': SumAggregation,
            'mean' : MeanAggregation,
            'sort': SortAggregation,
            'set2set': Set2Set,
            #'att' : AttentionalAggregation
        }
    
    def factory(self, type, **kwargs):
        try:
            fun = self.aggr_types[type]
            return fun(**kwargs)
        except KeyError:
            raise ValueError(f"Invalid aggregation type: {type}")
        except TypeError:
            raise ValueError(f"Invalid arguments for aggregation type: {type}")
    
    def dl_in_channels_factor(self, type, **kwargs):
        if type == "sum":
            return 1
        elif type == "mean":
            return 1
        elif type == "sort":
            return kwargs["k"]
        elif type == "set2set":
            return 2 
        #elif type == "att":
        # pass
        else:
            raise ValueError(f"Invalid aggregation type: {type}")