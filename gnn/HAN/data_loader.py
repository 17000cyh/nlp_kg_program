"""
data is saved in ACM3025.mat

instead of using original graph, this data set use processed meta-path data to express relationship between different nodes

there are three adjacent matrix in data set: ptp, plp and pap, respectively represent nodes linked by
 meta-path Paper-keyword-Paper, Paper-Conference(maybe)-Paper and Paper-author-Paper

plp and pap are more important because the value of ptp is almost equaling to 1, so the matrix will be meaningless


"""

import scipy.io as scio
import torch
import numpy as np



def load_data():
    data = scio.loadmat("data/ACM3025.mat")

    adjs = {
        "PLP": torch.Tensor(data["PLP"]),
        "PAP": torch.Tensor(data["PAP"])
    }

    metapaths = adjs.keys()
    features = torch.Tensor(data["feature"])
    label = torch.argmax(torch.Tensor(data["label"]),dim = 1)

    train = data["train_idx"].reshape(-1)
    val = data["val_idx"].reshape(-1)
    test = data["test_idx"].reshape(-1)

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    train_idx = torch.LongTensor(train)
    val_idx = torch.LongTensor(val)
    test_idx = torch.LongTensor(test)

    return features, label, adjs, metapaths, train_idx, val_idx, test_idx



if __name__ == "__main__":
    features, label, adjs, metapaths, train_idx, val_idx, test_idx = load_data()

    print(label.shape)
    for item in label:
        print(item)

    print(label[train_idx])

    print(train_idx)

    print(test_idx.shape)