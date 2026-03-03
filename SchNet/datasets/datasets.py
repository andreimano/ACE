import os.path as osp

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose


class GetLabelIdx(BaseTransform):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, data):
        data.y = data.y[:, self.idx]
        return data


def get_dataset(args):
    if hasattr(args, "dataset") and args.dataset != "qm9":
        raise ValueError(f"Only QM9 is supported, got dataset={args.dataset!r}")

    print("Loading QM9 dataset...")
    print(f"Training on target property index {args.target_idx} ...")

    path = osp.join(osp.dirname(osp.realpath(__file__)), args.data_dir)
    transforms = Compose([GetLabelIdx(args.target_idx)])
    dataset = QM9(path, transform=transforms).shuffle()

    train_ratio = 0.8
    val_ratio = 0.1
    n = len(dataset)
    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)

    train_dataset = dataset[:train_count]
    val_dataset = dataset[train_count: train_count + val_count]
    test_dataset = dataset[train_count + val_count:]

    print(f"Loaded QM9 dataset with {len(dataset)} molecules.")
    print("Split sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
