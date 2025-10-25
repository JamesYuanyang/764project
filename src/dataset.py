import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

class TwoClassWrapper(Dataset):
    def __init__(self, base, cls_a, cls_b, max_samples=None):
        self.base = base
        idx = [i for i, y in enumerate(base.targets) if y in (cls_a, cls_b)]
        if max_samples: random.shuffle(idx); idx = idx[:max_samples]
        self.indices = idx
        self.cls_a, self.cls_b = cls_a, cls_b
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        y = 0 if y == self.cls_a else 1
        return x, y

def build_tfm(mean, std):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def load_task_loaders(cfg_dataset, cfg_dl):
    root = cfg_dataset["root"]
    mean = cfg_dataset.get("transforms", {}).get("normalize_mean", [0.4914,0.4822,0.4465])
    std  = cfg_dataset.get("transforms", {}).get("normalize_std", [0.2023,0.1994,0.2010])
    tfm = build_tfm(mean, std)

    trainset = datasets.CIFAR10(root=root, train=True, transform=tfm, download=False)
    testset  = datasets.CIFAR10(root=root, train=False, transform=tfm, download=False)

    pairs = cfg_dataset["class_pairs"]
    sizes = cfg_dataset["sample_sizes"]

    train_loaders, val_loaders = [], []
    for (a,b), n in zip(pairs, sizes):
        ds_tr = TwoClassWrapper(trainset, a, b, max_samples=n)
        ds_te = TwoClassWrapper(testset, a, b)
        train_loaders.append(DataLoader(ds_tr, batch_size=cfg_dl["batch_size"], shuffle=True))
        val_loaders.append(DataLoader(ds_te, batch_size=cfg_dl["batch_size"], shuffle=False))
        print(f"Task {a}-{b}: train={len(ds_tr)}, val={len(ds_te)}")
    return train_loaders, val_loaders
