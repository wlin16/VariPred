import config

from tqdm import tqdm
import pickle
import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Data process part
def get_truncation(chop):

    chop.reset_index(drop=True, inplace=True)

    print(' The amount of sequences need to be truncated: ', len(chop))

    for index, seq in tqdm(chop.iterrows(), total=chop.shape[0]):

        if seq['aa_index'] < 1022:
            select_wt = seq['wt_seq'][0:1022]
            select_mt = seq['mt_seq'][0:1022]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = seq['aa_index']

        elif seq['aa_index'] > seq['Length'] - 1022:
            select_wt = seq['wt_seq'][-1022:]
            select_mt = seq['mt_seq'][-1022:]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = seq['aa_index']-seq['Length']+1022
        else:
            select_wt = seq['wt_seq'][seq['aa_index'] -
                                      511:seq['aa_index'] + 511]
            select_mt = seq['mt_seq'][seq['aa_index'] -
                                      511:seq['aa_index'] + 511]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = 511
    chop["new_index"] = chop["new_index"].astype(int)

    return chop


def df_process(df):
    remain_df = df[df['Length'] <= 1022]
    trunc_df = df[df['Length'] > 1022]

    remain_df['new_index'] = remain_df['aa_index']

    truncated_df = get_truncation(trunc_df)

    truncated_result = pd.concat(
        [remain_df, truncated_df]).reset_index(drop=True)

    return truncated_result


class ESMDataset(Dataset):
    def __init__(self,row, datatype):
        super().__init__()
        self.seq = row[f'{datatype}_seq']
        self.aa = row[f'{datatype}_aa']
        self.gene_id = row['record_id']
        self.aa_index = row['new_index']
        self.label = row['label']
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        return (self.label[idx],self.seq[idx],self.aa[idx],self.gene_id[idx],self.aa_index[idx])

        
def collate_fn(batch):
    labels, sequences, aa, gene_id, aa_index = zip(*batch)
    return list(zip(labels, sequences)), aa, gene_id, aa_index

def get_logits(total_logits,aa,esm_dict):
    softmax = nn.Softmax(dim=-1)
    aa_id = [esm_dict[x]-4 for x in aa]
    
    batch_aa_id = torch.arange(len(aa_id))
    logits = softmax(total_logits)[batch_aa_id, aa_id]
    return logits
    
def generate_embeds_and_save(df, save_path, data_class, model, batch_converter, alphabet, device=config.device):
    if os.path.exists(save_path + '/' + data_class + '.pt'):
        print(f'****** {data_class} embedding already exists ******')
    else:
        esm_dict = alphabet.tok_to_idx
        batch_converter = alphabet.get_batch_converter()
        model = model.to(device) # move your model to GPU

        wt_dataset = ESMDataset(df,datatype="wt")
        wt_dataloader = DataLoader(wt_dataset, batch_size=config.batch_size_for_embed_gen, shuffle=False,collate_fn=collate_fn, drop_last=False)
        mt_dataset = ESMDataset(df,datatype="mt")
        mt_dataloader = DataLoader(mt_dataset, batch_size=config.batch_size_for_embed_gen, shuffle=False,collate_fn=collate_fn, drop_last=False)

        label_for_embeds =[]
        gene_id_list = []
        concat = []
        logits_list = []

        for i,j in tqdm(zip(wt_dataloader,mt_dataloader),total=len(wt_dataloader)):
            batch_labels, _, wt_batch_tokens = batch_converter(i[0])
            _, _, mt_batch_tokens = batch_converter(j[0])
            wt_aa, wt_gene_id, wt_aa_index = i[1],i[2],i[3]
            mt_aa = j[1]
            label_for_embeds.append(batch_labels)
            gene_id_list.append(wt_gene_id)

            with torch.no_grad():
                aa_index = torch.tensor(wt_aa_index).to(device)
                batch_indices = torch.arange(len(aa_index))
                wt_result = model(wt_batch_tokens.to(device), repr_layers=[33])
                mt_result = model(mt_batch_tokens.to(device), repr_layers=[33])

                wt_repr = wt_result["representations"][33][batch_indices, aa_index] # fetch embeds
                mt_repr = mt_result["representations"][33][batch_indices, aa_index] # batch_size, embedding_size

                result = torch.cat((wt_repr, mt_repr), dim=1) # concat wt_emb with wt_emb -> batch_size, 2560

                total_logits = wt_result['logits'][:,:,4:24] # batch_size, max_seq_len, 20
                total_logits = total_logits[batch_indices, aa_index]

                wt_logits = get_logits(total_logits, wt_aa, esm_dict)
                mt_logits = get_logits(total_logits, mt_aa, esm_dict)
                logits = torch.log(mt_logits/wt_logits).unsqueeze(1)

                concat.append(result)
                logits_list.append(logits)

        concat = torch.cat(concat, dim=0).detach().cpu() # concat all embeddings together -> shape: length of the dataset, 2560
        logits_list=torch.cat(logits_list, dim=0).detach().cpu()
        gene_list = [str(x) for tup in gene_id_list for x in tup]
        label_list = [x for item in label_for_embeds for x in item]
        final_result = {'x': concat, 'label': label_list,'logits': logits_list, 'record_id': gene_list}

        if not os.path.isdir(f'{save_path}'):
            os.makedirs(f'{save_path}')  # create the dir for embeddings

        print(f"****** {data_class} embedding Saving path is: ",
              save_path, ' ******')

        save_path = save_path + '/' + data_class

        # save your embeddings
        torch.save(final_result,f'{save_path}.pt') # save the embeddings
    



# model training part:

# fetch the embeddings
def unpickler(ds_name):

    path = f'{config.esm_storage_path}/{ds_name}.pt'

    pt_embeds = torch.load(path, map_location=config.device)
    data_X = np.array(pt_embeds['x'].cpu())
    logits = np.array(pt_embeds['logits'].cpu()).reshape(-1, 1)
    
    data_y = pt_embeds['label']
    record_id = pt_embeds['record_id']

    data_X = np.hstack((data_X, logits))

    return data_X, data_y, record_id


# Prepare datasets for models
class VariPredDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]


# model architecture setup
class MLPClassifier_LeakyReLu(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """

    def __init__(self, num_input, num_hidden, num_output):
        super(MLPClassifier_LeakyReLu, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden = nn.Linear(num_input, num_hidden)
        self.predict = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_output)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
        x = self.softmax(x)

        return x


# train the model
def flat_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def trainer(train_loader, val_loader, model, device=config.device, early_stop=config.early_stop, n_epochs=config.n_epochs):

    criterion = nn.BCELoss(reduction='sum')  # Define the loss function

    # Define the optimization algorithm.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=0)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps= 0,
    #                                         num_training_steps= len(train_loader)*n_epochs)

    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        x = []
        for batch in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            # Move the data to device.
            b_seq, b_labels = tuple(t.to(device) for t in batch)

            pred = model(b_seq.float())
            b_labels = b_labels.float()
            loss = criterion(pred[:, 0], b_labels)

            # Compute gradient(backpropagation).
            loss.backward()

            optimizer.step()                    # Update parameters.
            # scheduler.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        ########### =========================== Evaluation=========================################
        print('\n\n###########=========================== Evaluating=========================################\n\n')

        model.eval()  # Set the model to evaluation mode.
        loss_record = []
        total_eval_accuracy = 0

        preds = []
        labels = []

        val_pbar = tqdm(val_loader, position=0, leave=True)
        for batch in val_pbar:

            # Move your data to device.
            b_seq, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                b_labels = b_labels.float()
                pred = model(b_seq.float())
                loss = criterion(pred[:, 0], b_labels)

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
                # Create directory of saving models.
                os.mkdir(f'{storage_path}')

            torch.save({
                'model_state_dict': model.state_dict(), },
                f'{storage_path}/model.ckpt')  # Save the best model

            print('\nSaving model with loss {:.3f}...'.format(best_loss))

            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')

            return


def predict(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode.
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        b_seq, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(b_seq.float())
            preds.append(pred[:, 0].detach().cpu())
            labels.append(b_labels.detach().cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    return preds, labels


def predict_results(y_true, preds, record_id, train=False, output_name=None):
    result_path = f'../example/output_results'  # path for saving predictions

    if not os.path.exists(f'{result_path}'):
        os.makedirs(result_path)

    if train:

        label_names = {'0': 0, '1': 1}

        auc_value = roc_auc_score(y_true, preds)
        print('AUC score: ', auc_value)

        y_true_np = np.array(y_true)
        preds = np.array(preds >= 0.2, dtype=int)

        MCC = matthews_corrcoef(y_true_np, preds)
        print('MCC: ', MCC)

        report = classification_report(
            y_true_np, preds, target_names=label_names)
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
