from model import HAN
from data_loader import load_data
from torch.optim import Adam
import torch
import torch.nn.functional as F

def train(epoch):
    features, label, adjs, metapaths, train_idx, val_idx, test_idx = load_data()

    model = HAN(features.shape[1],512,8,3,2)

    optimizer = Adam(model.parameters(),lr = 0.001)
    adjs_list = [adjs[key] for key in adjs.keys()]

    for _ in range(epoch):

        logits = model(features,adjs_list)

        prediction = F.log_softmax(logits,dim = 1)
        loss = F.nll_loss(prediction[train_idx],label[train_idx].long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        right_number = (torch.argmax(prediction,dim = 1)[train_idx] == label[train_idx]).sum()
        acc_train = right_number.item()/train_idx.shape[0]
        loss_train = loss.item()

        val_right_number = (torch.argmax(prediction,dim=1)[val_idx] == label[val_idx]).sum()
        acc_val = val_right_number.item()/val_idx.shape[0]
        loss_val = F.nll_loss(prediction[val_idx],label[val_idx].long()).item()

        print("\nInformation about %s epoch"%_)
        print("train acc is %s and train loss is %s"%(acc_train,loss_train))
        print("val acc is %s and val loss is %s"%(acc_val,loss_val))

        if _ %5 == 0:
            test_right_number = (torch.argmax(prediction, dim=1)[test_idx] == label[test_idx]).sum()
            acc_test = test_right_number.item() / test_idx.shape[0]
            loss_test = F.nll_loss(prediction[test_idx], label[test_idx].long()).item()
            print("Test acc is %s and val loss is %s"%(acc_test,loss_test))


train(500)





