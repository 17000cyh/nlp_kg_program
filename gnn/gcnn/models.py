import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy

class GCNN(nn.Module):
    def __init__(self, input_feature_number,hidden_feature_number,class_number,jump_number = 2):
        super(GCNN, self).__init__()
        self.jump_number = jump_number
        # 首先创建一个转移矩阵，将N*F的原始矩阵按照F*F'转移到N*F'当中
        # Create a transform matrix to transform origin feature matrix to a new space
        # print("The input feature number is %s\n hidden feature number is %s\n"%(input_feature_number,hidden_feature_number))
        self.feature_trans = nn.Parameter(torch.empty(size = (input_feature_number,hidden_feature_number)))
        nn.init.xavier_uniform_(self.feature_trans.data, gain=1.414)
        # print("The shape of feature trans is:")
        # print(self.feature_trans.shape)
        self.W = nn.Parameter(torch.empty(size = (hidden_feature_number,jump_number)))

        # 接下来创建全连接层
        # Create a dense layer
        self.dense = nn.Parameter(torch.empty(size=(class_number,hidden_feature_number*jump_number)))
        nn.init.xavier_uniform_(self.dense.data, gain=1.414)


    def forward(self, x, adj):
        x = x.matmul(self.feature_trans)
        adj_list = []
        for i in range(self.jump_number):
            temp_tensor = adj
            for j in range(i):
                temp_tensor = temp_tensor.matmul(adj)
            adj_list.append(temp_tensor.matmul(x))

        Z = torch.stack(adj_list,0)
        logits = torch.stack(
            [self.dense.matmul( Z[:,i,:].reshape(-1)) for i in range(Z.shape[1])],
            0
        )
        return logits


if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GCNN(features.shape[1],100,7,jump_number=2)
    optimizer = optim.Adam(model.parameters(),
                           lr= 0.05,)


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_val.data.item()

    for i in range(0,100):
        train(i)