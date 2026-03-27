"""Unified dataset splitting and persona grouping."""

from collections import defaultdict

from datasets import load_dataset, Dataset

from bayesft.utils.config import DEFAULT_SPLIT_NAMES, DEFAULT_SPLIT_SIZES


class DataSplitter:
    """Split a HuggingFace dataset into named subsets grouped by prompt."""

    def __init__(
        self,
        dataset_name,
        data_split="train",
        split_names=None,
        split_sizes=None,
        output_dir="./data",
    ):
        self.split_names = split_names or DEFAULT_SPLIT_NAMES
        self.split_sizes = split_sizes or DEFAULT_SPLIT_SIZES
        self.paths = [f"{output_dir}/{name}" for name in self.split_names]

        print(f"> Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split=data_split).shuffle()

    def split_data(self):
        """Group by prompt, split into named subsets, and save to disk."""
        prompts = group_by_column(self.dataset, "prompt")
        print("> Splitting data")
        keys = list(prompts.keys())
        n = len(keys)
        idxs = [int(sum(self.split_sizes[: i + 1]) * n) for i in range(len(self.split_sizes))]

        splits = []
        start = 0
        for end in idxs:
            splits.append([prompts[keys[i]] for i in range(start, end)])
            start = end

        self.splits = splits
        self.data_map = {name: split for name, split in zip(self.split_names, splits)}

        print("> Saving splits")
        for split, path in zip(splits, self.paths):
            _save_split(split, path)

        return self


def group_by_column(dataset, column):
    """Group dataset rows by a column value."""
    group = defaultdict(list)
    for row in dataset:
        group[row[column]].append(row)
    return dict(group)


def group_and_save_personas(sft_path, output_dir="./data/sft_personas"):
    """
    Load SFT dataset, group by persona, and save each persona's data separately.

    Returns:
        Number of unique personas found.
    """
    from datasets import load_from_disk

    ds = load_from_disk(sft_path)
    personas = group_by_column(ds, "persona")

    print(f"Found {len(personas)} unique personas, saving to {output_dir}")
    for idx, key in enumerate(personas.keys()):
        rows = personas[key]
        out_ds = Dataset.from_dict(
            {
                "prompt": [r["prompt"] for r in rows],
                "completion": [r["completion"] for r in rows],
                "persona": [r["persona"] for r in rows],
            }
        )
        out_ds.save_to_disk(f"{output_dir}/{idx}")

    return len(personas)


def get_unique_personas(dataset):
    """Extract unique persona strings from a dataset."""
    return list(set(dataset["persona"]))


def _save_split(split_data, path):
    """Flatten nested split data and save as HuggingFace Dataset."""
    prompts = []
    completions = []
    personas = []

    for group in split_data:
        for row in group:
            prompts.append(row["prompt"])
            completions.append(row.get("completion", row.get("completions", "")))
            personas.append(row["persona"])

    out_ds = Dataset.from_dict(
        {"prompt": prompts, "persona": personas, "completion": completions}
    )
    out_ds.save_to_disk(path)
