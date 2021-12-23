"""
This model are trying to implement HAN, a heterogeneous graph attention network

"""

import torch.nn as nn

import torch.nn.functional as F
from layer import NodeAttention,SemanticAttention

class HAN(nn.Module):

    def __init__(self,input_features,n_hid,head_number,class_number,meta_path_number,dropout = 0.5,alpha = 0.5):
        super(HAN, self).__init__()
        self.node_attention = NodeAttention(input_features,n_hid,head_number,dropout,alpha,meta_path_number)

        self.semantic_attention = SemanticAttention(n_hid,n_hid)

        self.dropout = dropout

        self.classifier = nn.Linear(n_hid,class_number)

    def forward(self,features,adjs):

        Z = self.semantic_attention(self.node_attention(features,adjs))

        Z = F.dropout(Z,self.dropout,training=self.training)

        result = self.classifier(Z)

        return result



