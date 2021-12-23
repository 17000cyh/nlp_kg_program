import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file will implement the function of Graph attention net work.

Abstractly, the function of GAT is to get an input of [N,feature_numbers] and output a tensor of [N,label_numbers]
An important util of GAT is GraphAttentionLayer, whose process is as follow:

1.transform the input with dimension feature_numbers to out feature
2.compute the attention scores and compute softmax result 
3.sum up

The computation of attention is simply inner-product of a vector a and [Wh_i||Wh_j]
"""
class GraphAttentionLayer(nn.Module):
    """
    This is a simple version of Graph attention layer
    """
    def __init__(self,input_features,output_features,dropout,alpha,concat = True):
        super(GraphAttentionLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.dropout = dropout
        self.alpha = alpha

        self.trans = nn.Parameter(torch.empty(size = (input_features,output_features)))
        # This Linear Module can transform a matrix of [batch_size,input_features] to [batch_size,output_features]
        nn.init.xavier_uniform_(self.trans)
        # Initialize the value of transform matrix

        self.attention = nn.Parameter(torch.empty(size = (2*output_features,1)))
        nn.init.xavier_uniform_(self.attention)
        # attention util is created

        self.leakyRelu = nn.LeakyReLU(self.alpha)
        self.concat = concat

    def forward(self,x,adj):
        """
        We can split attention scores between node i and node j into two part:

        Consider that the attention scores is computed by vector a and a concat vector Wh_i and Wh_j, scores_{ij} = a[Wh_i||Wh_j],
        scores_{ij} = a[:output_features] * Wh_i + a[:output_features] * Wh_j

        So, we can compute the two part and use e_i and e_j to denote them.
        """
        x = x.matmul(self.trans)
        # transform x as a [batch_size,output_features] tensor

        e_1 = x.matmul(self.attention[:self.output_features,:])
        e_2 = x.matmul(self.attention[self.output_features:,:])

        attention_scores = self.leakyRelu(e_1 + e_2.T)


        masked_attention_scores = attention_scores.masked_fill_(adj == 0,-1e15)
        # masked item whose value is zero so the value will be zero after softmax
        # it means that target node will not pay any attention to these nodes

        attention = F.softmax(masked_attention_scores,dim = 1)
        # remember to set the dimension equals to 1
        attention = F.dropout(attention,self.dropout,training= self.training)

        if self.concat:
            return F.elu(attention.matmul(x))
        else:
            return attention.matmul(x)

class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.first_layer = nn.ModuleList([GraphAttentionLayer(nfeat,nhid,dropout,alpha) for _ in range(nheads)])

        self.output_layer = GraphAttentionLayer(nheads*nhid,nclass,dropout,alpha)

    def forward(self,x,adj):
        x = F.dropout(x,self.dropout,training = self.training)
        x = torch.cat([attention(x,adj) for attention in self.first_layer],dim = 1)
        # pay attention that you must set the dimension as 1 here.

        x =  F.dropout(x,self.dropout,training = self.training)
        x = F.elu(self.output_layer(x,adj))
        return F.log_softmax(x, dim = 1)



if __name__ == "__main__":
    '''
    features = torch.randn((3,10))

    input_features = features.shape[1]

    mask = torch.Tensor([[1,1,0],[1,1,0],[1,0,1]])

    model = GAT(input_features,8,5,0.1,0.5,4)

    result = model(features,mask)

    print(result.shape)
    '''
    from utils import load_data,accuracy
    import torch.optim as optim
    import time

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    model = GAT(features.shape[1],32,7,0.2,0.5,4)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.005, )


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        print(output[idx_train][0])
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


    for i in range(0, 250):
        train(i)



