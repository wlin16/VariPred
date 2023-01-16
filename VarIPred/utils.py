import config

import csv
from tqdm import tqdm
import pickle
import os
from termcolor import colored
import math

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset


# Data process part
def get_truncation(chop):
    
    chop.reset_index(drop=True, inplace=True)
    
    print(' The amount of sequences need to be truncated: ', len(chop))
    
    for index, seq in tqdm(chop.iterrows(), total=chop.shape[0]):
        
        if seq['aa_index'] < 1022:
            select_wt = seq['wt_seq'][0:1022]
            select_mt = seq['mt_seq'][0:1022]
            chop.loc[index,'wt_seq'] = select_wt
            chop.loc[index,'mt_seq'] = select_mt
            chop['new_index'] = seq['aa_index']

        elif seq['aa_index'] > seq['Length']- 1022:
            select_wt = seq['wt_seq'][-1022:]
            select_mt = seq['mt_seq'][-1022:]
            chop.loc[index,'wt_seq'] = select_wt
            chop.loc[index,'mt_seq'] = select_mt
            chop['new_index'] = seq['aa_index']-seq['Length']+1022
        else:
            select_wt = seq['wt_seq'][seq['aa_index'] - 511:seq['aa_index'] + 511]
            select_mt = seq['mt_seq'][seq['aa_index'] - 511:seq['aa_index'] + 511]
            chop.loc[index,'wt_seq'] = select_wt
            chop.loc[index,'mt_seq'] = select_mt
            chop['new_index'] = 511
    
    return chop
        
def df_process(df):
    remain_df = df[df['Length'] <= 1022]
    trunc_df = df[df['Length'] > 1022]
    
    remain_df['new_index'] = remain_df['aa_index']
    
    truncated_df = get_truncation(trunc_df)
    
    truncated_result = pd.concat([remain_df, truncated_df]).reset_index(drop = True)
    
    
    return truncated_result

   
def get_embeds_and_logits(raw_df,save_path, data_class, model, batch_converter, alphabet, device=config.device):   
    
    # limit the length of each sequence within 1022, as this is the maximum length allowed in ESM
    raw_df['Length'] = raw_df['wt_seq'].apply(lambda x: len(x))

    if max(set(raw_df.Length)) > 1022:
        raw_df = df_process(raw_df)
    else: 
        raw_df['new_index'] = raw_df['aa_index']
        
    print('data length:', raw_df.shape[0])
    
    raw_df['record_id'] = raw_df['target_id']
     
    xs = []
    result = None
    
    softmax = nn.Softmax(dim=0)
    
    for index, seq in tqdm(raw_df.iterrows(), total=raw_df.shape[0]):
        s_len = len(seq['wt_seq'].replace(" ",'')) + 1
        aa_index = seq['new_index']
        old_aa_index = seq['aa_index']
        label = seq['label']
        wt_aa = seq['wt_aa']
        mt_aa = seq['mt_aa']
        wt_seq = seq['wt_seq']
        mt_seq = seq['mt_seq']
        gene_id = seq['record_id']


        # transform the data into ESM dataset
        wt_tuple = [(label, wt_seq)]
        mt_tuple = [(label, mt_seq)]


        wt_batch_labels, wt_batch_strs, wt_batch_tokens = batch_converter(wt_tuple)
        mt_batch_labels, mt_batch_strs, mt_batch_tokens = batch_converter(mt_tuple)

        
        with torch.no_grad():
            
            model = model.to(device)
            wt_repr = model(wt_batch_tokens.to(device), repr_layers=[33])["representations"][33] 
            wt_aa_repr = wt_repr[:, aa_index, :]
            total_logits = model(wt_batch_tokens.to(device), repr_layers=[33])['logits'][0][aa_index][4:24] # get the probabilities of 20 aa at the target position
            wt_aa_id = alphabet.tok_to_idx[wt_aa]-4 # align the aa index with the dictionary used in the pretraining process
            
            wt_logits = softmax(total_logits)[wt_aa_id] #get the probabilities of the wildtype aa

            mt_repr = model(mt_batch_tokens.cuda(), repr_layers=[33])["representations"][33]
            mt_aa_repr = mt_repr[:, aa_index, :]
            mt_aa_id = alphabet.tok_to_idx[mt_aa]-4
            
            mt_logits = softmax(total_logits)[mt_aa_id] #get the probabilities of the mutant type aa
            
            # calculate the log likelihood
            logits = math.log(mt_logits/wt_logits)


        wt_emb = wt_aa_repr.detach().cpu().numpy().squeeze()
        mt_emb = mt_aa_repr.detach().cpu().numpy().squeeze()
        
        concate = np.concatenate((wt_emb, mt_emb))
        xs.append({'x':concate.reshape(1,-1),'label':label,'logits':logits,'record_id':gene_id})
    
    if not os.path.isdir(f'{save_path}'):
        os.makedirs(f'{save_path}')  # create the dir for embeddings
    
    save_path = save_path + '/' + data_class
    
    print(f"****** {data_class} embedding Saving path is: ", save_path, ' ******') 
    
    with open(f'{save_path}.pkl', 'wb') as f:
        pickle.dump(xs, f) # ouput file e.g. /example/embeds/train.pkl
        
        
        


# model training part:

## fetch the embeddings
def unpickler(ds_name):

    path = f'{config.esm_storage_path}/{ds_name}.pkl'

    concat=[]
    with open(path, 'rb') as file:
        y = pickle.load(file)
        concat = y

        data_y = []
        data_X = []
        logits = []
        record_id = []
        for i in range(len(concat)):
            data_X.append(concat[i]['x'][0])
            logits.append(concat[i]['logits'])
            data_y.append(concat[i]['label'])
            record_id.append(concat[i]['record_id'])

        data_X = np.array(data_X)
        logits = np.array(logits).reshape(-1,1)
        
        model_size = 1280

        data_X  = np.hstack((data_X, logits))
        
        return data_X, data_y, record_id



## Prepare datasets for models 
class VarIPredDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)
    
    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]


## model architecture setup
class MLPClassifier_LeakyReLu(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """
    def __init__(self,num_input,num_hidden,num_output):
        super(MLPClassifier_LeakyReLu, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden=nn.Linear(num_input,num_hidden)
        self.predict = nn.Sequential(
            nn.Dropout(0.5), 
            nn.LeakyReLU(inplace = True),
            nn.Linear(num_hidden, num_output)
        )
        self.softmax = nn.Softmax(dim=1)
           
    def forward(self,x):
        x=self.hidden(x)
        x=self.predict(x)
        x = self.softmax(x)

        return x


## train the model
def flat_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def trainer(train_loader, val_loader, model, device = config.device, early_stop = config.early_stop, n_epochs = config.n_epochs):

    criterion = nn.BCELoss(reduction='sum') # Define the loss function

    # Define the optimization algorithm. 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay = 0)
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                         num_warmup_steps= 0,
    #                                         num_training_steps= len(train_loader)*n_epochs)

    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop
    
    for epoch in range(n_epochs):
        model.train() # Set the model to train mode.
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        x=[]
        for batch in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            b_seq, b_labels = tuple(t.to(device) for t in batch) # Move the data to device. 

            pred = model(b_seq.float())  
            b_labels = b_labels.float()
            loss = criterion(pred[:,0], b_labels)
            
            loss.backward()                     # Compute gradient(backpropagation).
            
            optimizer.step()                    # Update parameters.
            # scheduler.step()
            
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        ###########=========================== Evaluation=========================################
        print('\n\n###########=========================== Evaluating=========================################\n\n')

        model.eval() # Set the model to evaluation mode.
        loss_record = []
        total_eval_accuracy = 0
        
        preds = []
        labels = []
        
        val_pbar = tqdm(val_loader, position=0, leave=True)
        for batch in val_pbar:
            
            b_seq, b_labels = tuple(t.to(device) for t in batch) # Move your data to device. 
            with torch.no_grad():
                b_labels = b_labels.float()
                pred = model(b_seq.float())
                loss = criterion(pred[:,0], b_labels)
                
                # preds.append(pred[:,0].detach().cpu()[0].tolist()) 
                # labels.append(b_labels.detach().cpu()[0].tolist())
                
            loss_record.append(loss.item())
            total_eval_accuracy += flat_accuracy(pred, b_labels)

            val_pbar.set_description(f'Evaluating [{epoch + 1}/{n_epochs}]')
            val_pbar.set_postfix({'evaluate loss': loss.detach().item()})
            
            
        # For selecting the best MCC threshold 
        # breakpoint()
        # y_true_np = np.array(labels)
        # pred_np = np.array(preds)
        # for label, pred_value in zip(y_true_np, pred_np):
        #     with open(f'./threhold_pick.txt', 'a+') as f:
        #         f.write(f'{label}\t{pred_value}\n')
        
        mean_valid_loss = sum(loss_record)/len(loss_record)
        avg_val_accuracy = total_eval_accuracy / len(val_loader)

        print(f'\nEpoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            
            storage_path = f'./model'
            
            if not os.path.isdir(f'{storage_path}'):
                os.mkdir(f'{storage_path}') # Create directory of saving models.
            
            torch.save({
                'model_state_dict': model.state_dict(),},
                       f'{storage_path}/model.ckpt') # Save the best model
            
            print('\nSaving model with loss {:.3f}...'.format(best_loss))
            
            
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            
            return

def predict(test_loader, model, device):
    model.eval() # Set the model to evaluation mode.
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        b_seq, b_labels = tuple(t.to(device) for t in batch)                   
        with torch.no_grad():                   
            pred = model(b_seq.float())        
            preds.append(pred[:,0].detach().cpu())
            labels.append(b_labels.detach().cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim = 0)

    return preds, labels


def predict_results(y_true, preds, record_id, train=False, output_name =None):
    result_path = f'../example/output_results'  # path for saving predictions

    if not os.path.exists(f'{result_path}'):
        os.makedirs(result_path)

    if train:

        label_names = {'0':0, '1':1}

        auc_value = roc_auc_score(y_true, preds)
        print('AUC score: ', auc_value)
        
        y_true_np = np.array(y_true)
        preds = np.array(preds >= 0.2, dtype=int)
        
        MCC = matthews_corrcoef(y_true_np, preds)
        print('MCC: ', MCC)
        
        report = classification_report(y_true_np, preds,target_names=label_names)
        print(report)
    
        # Saving the prediction results for each test data
        if not os.path.exists(f'{result_path}/model_eval_result.txt'):
            header = "target_id\tlabel\tprediction\n"
            with open(f'{result_path}/model_eval_result.txt', 'a') as file_writer:
                file_writer.write(header)

        for ids, label, pred_value in zip(record_id, y_true_np, preds):
            with open(f'{result_path}/model_eval_result.txt', 'a+') as f:
                f.write(f'{ids}\t{label}\t{pred_value}\n')
        
        with open(f'{result_path}/model_performance.txt', 'a') as file_writer:
                file_writer.write(f'MCC: {MCC}\nroc_auc_score: {auc_value}\n')

    else:
        
        preds = np.array(preds >= 0.2, dtype=int)
        
        if not os.path.exists(f'../example/{output_name}.txt'):
            header = "target_id\tprediction\n"
            with open(f'{result_path}/{output_name}.txt', 'a') as file_writer:
                file_writer.write(header)

        for ids, pred_value in zip(record_id, preds):
            with open(f'{result_path}/{output_name}.txt', 'a+') as f:
                f.write(f'{ids}\t{pred_value}\n')
    

