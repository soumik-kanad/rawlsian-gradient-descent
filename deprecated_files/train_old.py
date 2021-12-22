from simplenet import SimpleNet
from uciadult_dataset import UCIAdultDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation_metrics import equalised_odds,predictive_parity,demographic_parity
import fire

def sgd_update_function(param, grad, learning_rate):
    return param - learning_rate * grad

def train(
    method='sgd',
    batch_size = 256,
    epochs = 10,
    learning_rate = 0.1,
    device = 'cuda',
    rgd = False,
):
    #pytorch train pipeline
    train_dataset=UCIAdultDataset()
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataset=UCIAdultDataset(split='test')
    test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    model=SimpleNet(train_dataset.num_features,1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion=nn.BCELoss()
    criterion_noReduce=nn.BCELoss(reduction='none')
    for epoch in range(epochs):
        epoch_losses=[]
        correct=0
        total=0
        accuracy=0
        for i, batch in enumerate(train_dataloader):
            x=batch['x'].float().to(device)
            y=batch['y'].float().to(device)
            # print(y)
            y_pred=model(x).squeeze()

            if rgd:
                losses=criterion_noReduce(y_pred,y,)
                
            else:
                loss=criterion(y_pred,y)
                
                #Manual gradient descent
                model.zero_grad()
                loss.backward()

            
                with torch.no_grad():
                    for p in model.parameters():
                        new_val = sgd_update_function(p, p.grad, learning_rate)
                        p.copy_(new_val)

            # #Autograd
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()


            predicted=(y_pred > 0.5).float()            
            epoch_losses.append(loss.item())
            total+=y.size(0)
            correct+= (predicted==y).sum().item()
        print("Epoch: {} Loss: {} Accuracy: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total))
        prev_accuracy=accuracy
        accuracy=correct/total
        # Scheduling learning rate
        # if prev_accuracy<0.85 and accuracy > 0.85:
        #     learning_rate=learning_rate*0.5

        # print(y_pred,y)

    #test model
    correct=0
    total=0
    fp=0
    fn=0
    tp=0
    tn=0
    summ=0
    summ1=0
    Y_pred=[]
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x=batch['x'].float().to(device)
            y=batch['y'].float().to(device)
            # print(y)
            y_pred=model(x).squeeze()
            predicted=(y_pred > 0.5).float()
            summ+=predicted.sum().item()
            summ1+=y.sum().item()
            total+=y.size(0)
            correct+= (predicted==y).sum().item()
            fp+=torch.logical_and(predicted==1, y==0).sum().item()
            fn+=torch.logical_and(predicted==0, y==1).sum().item()    
            tp+=torch.logical_and(predicted==1, y==1).sum().item()
            tn+=torch.logical_and(predicted==0, y==0).sum().item()

            Y_pred=Y_pred+predicted.cpu().detach().numpy().tolist()
    print(summ,summ1,correct,total)
    print("fp=",fp,",fn=",fn,",tp=",tp,",tn=",tn)   


    print("Test Accuracy: {}".format(correct/total))

    
    sex_groups=test_dataset.discrete_columns['sex']
    eo_sex=equalised_odds(df=test_dataset.df, sensitive_arribute_col='sex', sensitive_arribute_groups=sex_groups, target_col='income', prediction=Y_pred)
    pp_sex=predictive_parity(df=test_dataset.df, sensitive_arribute_col='sex', sensitive_arribute_groups=sex_groups, target_col='income', prediction=Y_pred)
    dp_sex=demographic_parity(df=test_dataset.df, sensitive_arribute_col='sex', sensitive_arribute_groups=sex_groups, prediction=Y_pred)
    print('Sex:')
    print("Equalised Odds:")
    print(eo_sex)
    print("Predictive Parity:")
    print(pp_sex)
    print("Demographic Parity:")
    print(dp_sex)

    # 'race'['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    race_groups=test_dataset.discrete_columns['race']
    eo_race=equalised_odds(df=test_dataset.df, sensitive_arribute_col='race', sensitive_arribute_groups=race_groups, target_col='income', prediction=Y_pred)
    pp_race=predictive_parity(df=test_dataset.df, sensitive_arribute_col='race', sensitive_arribute_groups=race_groups, target_col='income', prediction=Y_pred)
    dp_race=demographic_parity(df=test_dataset.df, sensitive_arribute_col='race', sensitive_arribute_groups=race_groups, prediction=Y_pred)
    print('Race:')
    print("Equalised Odds:")
    print(eo_race)
    print("Predictive Parity:")
    print(pp_race)
    print("Demographic Parity:")
    print(dp_race)

if __name__=="__main__":
    fire.Fire(train)





# #normal way
# pred = model(inp)
# loss = critetion(pred, ground_truth)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# #manual way
# pred = model(inp)
# loss = your_loss(pred)
# model.zero_grad()
# loss.backward()
# with torch.no_grad():
#   for p in model.parameters():
#     new_val = update_function(p, p.grad, loss, other_params)
#     p.copy_(new_val)


# from sklearn.model_selection import KFold
# k_folds = 5
#     # Define the K-fold Cross Validator
#     kfold = KFold(n_splits=k_folds, shuffle=True)
#     for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#     # Print
#     print(f'FOLD {fold}')
#     print('--------------------------------')
#     # Sample elements randomly from a given list of ids, no replacement.
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)