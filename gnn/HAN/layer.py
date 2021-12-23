import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttentionPerMetaPath(nn.Module):
    """
    This class will implement the node-level attention for one meta-path
    """
    def __init__(self,input_features,output_features,dropout,alpha):
        super(NodeAttentionPerMetaPath, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha)

        self.trans = nn.Parameter(torch.empty(input_features,output_features))
        nn.init.xavier_uniform_(self.trans.data,1.414)

        self.attention = nn.Parameter(torch.empty(2* output_features,1))
        nn.init.xavier_uniform_(self.attention.data,1.414)


    def forward(self,x,mask):
        # mask is the adjacent matrix for this meta-path
        x = F.dropout(x,self.dropout,training = self.training)

        x = x.matmul(self.trans)
        # transform into [node_number,output_features]

        e_1 = x.matmul(self.attention[:self.output_features,:])
        e_2 = x.matmul(self.attention[self.output_features:,:])

        scores = self.leakyReLU(e_1 + e_2.T)

        scores = F.dropout(scores,self.dropout,training = self.training)

        masked_scores = scores.masked_fill_(mask == 0, -1e15)

        attention = F.softmax(masked_scores, dim = 1)

        return attention.matmul(x)

class MultiHeadNodeAttentionPerMetaPath(nn.Module):

    def __init__(self,input_features,output_features,dropout,alpha,head_number):
        super(MultiHeadNodeAttentionPerMetaPath, self).__init__()

        assert output_features % head_number == 0
        # make sure that features number for each head is an integer

        self.attentions =  nn.ModuleList([NodeAttentionPerMetaPath(input_features,output_features // head_number,dropout,alpha)
                                          for _ in range(head_number)])

    def forward(self,x,mask):
        return torch.cat([attention(x,mask) for attention in self.attentions],dim=1)



        
        
class NodeAttention(nn.Module):

    def __init__(self,input_features,output_features,head_number,dropout,alpha,metapath_number):
        super(NodeAttention, self).__init__()

        self.metapaths_attentions = nn.ModuleList([MultiHeadNodeAttentionPerMetaPath(input_features,output_features,dropout,alpha,head_number)
                                              for _ in range(metapath_number)])

        self.metapath_number = metapath_number

    def forward(self,x,adjs):
        # adjs is the mask for different meta path

        assert len(adjs) == self.metapath_number

        # return a [metapath_number,node_number,out_features] tensor
        return F.sigmoid(torch.stack([meta_path_attention(x,adjs[i]) for i,meta_path_attention in enumerate(self.metapaths_attentions)]))


class SemanticAttention(nn.Module):

    def __init__(self,input_features,output_features):
        super(SemanticAttention, self).__init__()

        self.W = nn.Parameter(torch.empty(input_features,output_features))
        nn.init.xavier_uniform_(self.W.data,1.414)

        self.b = nn.Parameter(torch.empty(1,output_features))
        nn.init.xavier_uniform_(self.b.data,1.414)

        self.q = nn.Parameter(torch.empty(output_features,1))
        nn.init.xavier_uniform_(self.q.data,1.414)


    def forward(self,node_attentions):
        # input is a [meta_path_number,node_number,input_features] tensor

        trans = F.tanh(node_attentions.matmul(self.W) + self.b)
        # trans is a [meta_path_number,node_number,output_features] tensor

        w_meta = trans.matmul(self.q).reshape(trans.shape[0],trans.shape[1])

        # w_meta is a [meta_path_number,node_number] tensor

        w_meta = w_meta.sum(dim = 1) / w_meta.shape[1]

        beta = F.softmax(w_meta)

        Z = torch.zeros(size = (node_attentions[0].shape))

        for i,weight in enumerate(beta):
            Z += weight * node_attentions[i]

        return Z


if __name__ == "__main__":
    from data_loader import load_data

    features, label, adjs, metapaths, train_idx, val_idx, test_idx = load_data()

    print(features)

    one_metapath_attention_test = NodeAttentionPerMetaPath(features.shape[1],300,0.1,0.5)
    print(adjs.keys())
    test_mask = adjs["PAP"]

    result = one_metapath_attention_test(features,test_mask)

    print(result.shape)
    print(result[0])

    node_attention_test = NodeAttention(features.shape[1],300,0.1,3,0.5,2)

    test_mask = [adjs[key] for key in adjs.keys()]

    result = node_attention_test(features,test_mask)

    print(result.shape)
    print(result[0])

    semantic_attention_test = SemanticAttention(300,300)

    result = semantic_attention_test(result)

    print(result.shape)

    print(result[0])






















