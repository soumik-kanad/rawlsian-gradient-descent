from simplenet import SimpleNet
from uciadult_dataset import UCIAdultDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from evaluation_metrics import equalised_odds,predictive_parity,demographic_parity
from torch.utils.tensorboard import SummaryWriter
import fire
import os
import time

def sgd_update_function(param, grad, learning_rate):
    return param - learning_rate * grad

def sgd_update_function_normalised(param, grad, learning_rate,batch_size,eps=1e-12):
    return param - learning_rate * grad/grad.norm().clamp_min(eps)/batch_size

def evaluate(test_dataloader,model,device,writer, sensitive_attrs=[]):  
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
    writer.add_scalar('test/accuracy', correct/total, 0)
    writer.add_scalar('test/fp', fp, 0)
    writer.add_scalar('test/fn', fn, 0)
    writer.add_scalar('test/tp', tp, 0)
    writer.add_scalar('test/tn', tn, 0)


    print("Test Accuracy: {}".format(correct/total))

    test_dataset=test_dataloader.dataset

    def df_to_dict(df):
        return {col:df[col].values for col in df.columns}

    if 'sex' in sensitive_attrs:
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
        #print(dict(zip(eo_sex.index, eo_sex.values)))
        # print(dict(zip(eo_sex.T.index, eo_sex.T.values)))
        # writer.add_scalars('test/sex/equalised_odds',df_to_dict(eo_sex),0)
        # writer.add_scalars('test/sex/predictive_parity',df_to_dict(pp_sex),0)
        # writer.add_scalars('test/sex/demographic_parity',df_to_dict(dp_sex),0)

    if 'race' in sensitive_attrs:
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
        # writer.add_scalars('test/race/equalised_odds',df_to_dict(eo_race),0)
        # writer.add_scalars('test/race/predictive_parity',df_to_dict(pp_race),0)
        # writer.add_scalars('test/race/demographic_parity',df_to_dict(dp_race),0)

def pretrain_func(
    train_dataloader,
    model,
    criterion,
    #fixed attributes
    batch_size = 256,
    epochs = 10,
    learning_rate = 0.1,
    device = 'cuda',
):
    print("Pretraining the model for {} epochs".format(epochs))
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    for epoch in range(epochs):
        epoch_losses=[]
        correct=0
        total=0
        accuracy=0
        start_time=time.time()
        for i, batch in enumerate(train_dataloader):
            x=batch['x'].float().to(device)
            y=batch['y'].float().to(device)
            y_pred=model(x).squeeze()
            loss=criterion(y_pred,y)
            
            # #Manual gradient descent
            # model.zero_grad()
            # loss.backward()

        
            # with torch.no_grad():
            #     for p in model.parameters():
            #         new_val = sgd_update_function(p, p.grad, learning_rate)
            #         p.copy_(new_val)

            #Autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            predicted=(y_pred > 0.5).float()            
            epoch_losses.append(loss.item())
            total+=y.size(0)
            correct+= (predicted==y).sum().item()
        
        # writer.add_scalar('train/loss', sum(epoch_losses)/len(epoch_losses), epoch)
        # writer.add_scalar('train/accuracy', correct/total, epoch)
        print("Epoch: {} Loss: {} Accuracy: {} Epoch Time: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total, time.time()-start_time))
    
    return model

def train(
    batch_size = 256,
    epochs = 10,
    learning_rate = 0.1,
    device = 'cuda',
    rgd = False,
    rgd_mode = 'both',
    rgd_k1=1,
    rgd_k2=1,
    rgd_lr1=None,
    rgd_lr2=None,
    sensitive_attr = 'sex',
    save_dir = 'experiments/default',
    pretrain = False,
    rgd_step1_remove_normalisation = False,
    train_data_balance=False,
    test_data_balance=False,
):

    if rgd and rgd_mode not in ['both', 'only1', 'only2']:
        raise ValueError("Incorrect RGD setting")
    if sensitive_attr not in ['sex', 'race']:
        raise ValueError("Incorrect sensitive attribute")

    #if rgd equations wise learning rate are not specified, use the same learning rate for both equations
    if rgd and (not rgd_lr1):
        rgd_lr1=learning_rate
    if rgd and (not rgd_lr2):
        rgd_lr2=learning_rate
    
    #pytorch train pipeline
    train_dataset=UCIAdultDataset(split='train',balance_target_classes=train_data_balance)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataset=UCIAdultDataset(split='test',balance_target_classes=test_data_balance)
    test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    # print(train_dataset.num_features) -> 90 without "Non-white"
    # quit()
    model=SimpleNet(train_dataset.num_features,1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion=nn.BCELoss()
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    criterion_noReduce=nn.BCELoss(reduction='none')


    if pretrain:
        model=pretrain_func(train_dataloader,model,criterion)
        #evaluate(test_dataloader,model,device,sensitive_attrs=[sensitive_attr],writer=writer)
        #quit()

    for epoch in range(epochs):
        epoch_losses=[]
        epoch_losses1=[]
        epoch_losses2=[]
        correct=0
        total=0
        accuracy=0
        start_time=time.time()
        for i, batch in enumerate(train_dataloader):
            x=batch['x'].float().to(device)
            y=batch['y'].float().to(device)
            # print(y)
            

            if rgd:
                loss=0
                if rgd_mode == 'both' or rgd_mode == 'only1':
                    loss1=0
                    for k in range(rgd_k1):
                        y_pred=model(x).squeeze()
                        losses=criterion_noReduce(y_pred,y,)
                        # print(criterion(y_pred,y))
                        
                        # print(losses.shape) #=>torch.Size([256])

                        #TODO: should I do step one without the normalisation?
                        # Principle 1
                        if not rgd_step1_remove_normalisation: #normalise
                            for ii in range(len(losses)):
                                model.zero_grad()
                                losses[ii].backward(retain_graph=True)
                                with torch.no_grad():
                                    for p in model.parameters():
                                        new_val = sgd_update_function_normalised(p, p.grad, rgd_lr1,batch_size=len(losses))
                                        p.data = new_val
                        else:#don't normalise
                            _loss=criterion(y_pred,y)
                            model.zero_grad()
                            _loss.backward()
                            with torch.no_grad():
                                for p in model.parameters():
                                    new_val = sgd_update_function(p, p.grad, rgd_lr1)
                                    p.data = new_val
                        
                        # print(losses)
                        
                        loss1+=torch.mean(losses)
                        # print('loss1:',loss1)
                        
                    epoch_losses1.append(loss1.item())
                    loss+=loss1

                #TODO: should these be done in 2 different steps?
                if rgd_mode == 'both' or rgd_mode == 'only2':
                    loss2=0
                    for k in range(rgd_k2):
                        y_pred=model(x).squeeze()
                        losses=criterion_noReduce(y_pred,y,)
                        # Principle 2
                        _start, _end = train_dataset.col_pos[sensitive_attr]
                        batch_sensitive_attr_one_hot = x[:, _start:_end]

                        #find the worst off group
                        max_grp_idx = -1
                        max_grp_loss = -float("Inf")
                        with torch.no_grad():
                            for kk in range(_end-_start): # find group losses for all groups in a sensitive attribute
                                if torch.sum(batch_sensitive_attr_one_hot[:,kk]).item() > 0: # if there are any samples in this group
                                    grp_loss=torch.mean(batch_sensitive_attr_one_hot[:,kk]*losses,dim=-1)
                                    # print(kk,grp_loss)
                                    
                                    if grp_loss > max_grp_loss:
                                        max_grp_idx = kk
                                        max_grp_loss = grp_loss
                            # print()
                        
                        # update the worst off group
                        # print(batch_sensitive_attr_one_hot[:,max_grp_idx].shape,losses.shape)
                        # print(batch_sensitive_attr_one_hot[:,max_grp_idx])
                        # print(losses)
                        worst_off_group_loss=torch.mean(batch_sensitive_attr_one_hot[:,max_grp_idx]*losses,dim=-1)
                        model.zero_grad()
                        worst_off_group_loss.backward()
                        #print(worst_off_group_loss)

                        with torch.no_grad():
                            for p in model.parameters():
                                new_val = sgd_update_function(p, p.grad, rgd_lr2)
                                p.copy_(new_val)
                        
                        loss2+=worst_off_group_loss
                    epoch_losses2.append(loss2.item())
                    loss+=loss2
                    
                


                            
            else:
                y_pred=model(x).squeeze()
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
        
        writer.add_scalar('train/loss', sum(epoch_losses)/len(epoch_losses), epoch)
        writer.add_scalar('train/accuracy', correct/total, epoch)
        if rgd:
            if rgd_mode == 'only1':
                writer.add_scalar('train/loss1',sum(epoch_losses1)/len(epoch_losses1),epoch)
                print("Epoch: {} Loss: {} Accuracy: {} Epoch Time: {} Loss_1: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total, time.time()-start_time, sum(epoch_losses1)/len(epoch_losses1)))
        
            elif rgd_mode == 'only2':
                writer.add_scalar('train/loss2',sum(epoch_losses2)/len(epoch_losses2),epoch)
                print("Epoch: {} Loss: {} Accuracy: {} Epoch Time: {} Loss_2: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total, time.time()-start_time, sum(epoch_losses2)/len(epoch_losses2)))
            else:
                writer.add_scalar('train/loss1',sum(epoch_losses1)/len(epoch_losses1),epoch)
                writer.add_scalar('train/loss2',sum(epoch_losses2)/len(epoch_losses2),epoch)
                print("Epoch: {} Loss: {} Accuracy: {} Epoch Time: {} Loss_1: {} Loss_2: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total, time.time()-start_time, sum(epoch_losses1)/len(epoch_losses1), sum(epoch_losses2)/len(epoch_losses2)))
        
        else:#sgd
            print("Epoch: {} Loss: {} Accuracy: {} Epoch Time: {}".format(epoch, sum(epoch_losses)/len(epoch_losses), correct/total, time.time()-start_time))
        
        prev_accuracy=accuracy
        accuracy=correct/total
        # Scheduling learning rate
        # if prev_accuracy<0.85 and accuracy > 0.85:
        #     learning_rate=learning_rate*0.5

        # print(y_pred,y)
    
    evaluate(test_dataloader,model,device,sensitive_attrs=[sensitive_attr],writer=writer)


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