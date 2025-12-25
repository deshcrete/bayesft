from dataset import DataSplitter
import persona

dataSplitter = DataSplitter(
    dataset_name = "desh2806/emgMisalgGenCompletions-Large",
    data_split = "train",
    split_names = ["sft", "infer", "mixture", "personas"],
    split_sizes = [0.1, 0.1, 0.2, 0.6]
)



