import utils
import config
import pandas as pd
import re
import os
import sys
import argparse
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import esm


def get_embeds(df, dataset):
    '''

    input: dataframe (e.g. ../example/dataset/train.csv)
    output: .pkl (e.g. ../example/embeds/train.pkl)

    '''

    model_1b, alphabet_1b = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter_1b = alphabet_1b.get_batch_converter()
    utils.get_embeds_and_logits(df, save_path = config.esm_storage_path, data_class=dataset, model = model_1b, batch_converter = batch_converter_1b, alphabet = alphabet_1b)


def train_VarIPred(train_ds, test_ds, valid_ds=None):

    '''

    input: 
        train_ds: path of embeddings of training set (e.g. ../example/embeds/train.pkl)
        test_ds: path of embeddings of test set
        valid_ds: path of embeddings of valid set

    output:
        model weights (./model/model.ckpt)
        model predictions (e.g. ../example/output_results/model_eval_result.txt)
        model performance (MCC, AUC) (e.g. ../example/output_results/model_performance.txt)

    '''

    X_train, y_train, _ = utils.unpickler(ds_name=train_ds)
    X_test, y_test, record_id = utils.unpickler(ds_name=test_ds)

    if valid_ds is not None:
        X_valid, y_valid = utils.unpickler(ds_name=valid_ds)
        print('valid set name: ', valid_ds)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            stratify=y_train,
                                                            random_state=config.SEED)

    
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)
    print('X_valid shape: ', X_valid.shape)
    
    train_dataset = utils.VarIPredDataset(X_train, y_train)
    val_dataset = utils.VarIPredDataset(X_valid, y_valid)
    test_dataset = utils.VarIPredDataset(X_test, y_test)

    #> Feed dataset to dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    
    print('\n\n\n\n')
    print('=============== VarIPred model training start ===============')
    
    model_size = X_train.shape[1]
    num_hidden = int(model_size/2) 
    
    print('model_size: ', model_size)
    print('num_hidden: ', num_hidden)
    
    model = utils.MLPClassifier_LeakyReLu(num_input = model_size, num_hidden = num_hidden, num_output = config.label_num).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    utils.trainer(train_loader, val_loader, model)


    print('=============== Predicting & Evaluating the trained model ===============')
    storage_path = f'./model'
    checkpoint=torch.load(f'{storage_path}/model.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, y_true = utils.predict(test_loader, model, config.device)
            
    utils.predict_results(y_true, preds,
                          model_selection,
                          record_id = record_id,
                          train = train
                          )


def run_VarIPred(target_ds,output):

    '''

    input: 
        target_ds: path of embeddings of target set (without a true label)

    output:
        model predictions (e.g. ../example/output_results/VarIPred_output.txt)

    '''


    X_target, y_target, record_id = utils.unpickler(ds_name=target_ds)
    print('X_target shape: ', X_target.shape)

    target_dataset = utils.VarIPredDataset(X_target, y_test)
    target_loader = DataLoader(target_dataset, batch_size=config.batch_size)

    model_size = X_target.shape[1]
    num_hidden = int(model_size/2) 
    model = utils.MLPClassifier_LeakyReLu(num_input = model_size, num_hidden = num_hidden, num_output = config.label_num).to(config.device)

    storage_path = f'./model'
    checkpoint=torch.load(f'{storage_path}/model.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    preds, y_true = utils.predict(target_loader, model, config.device)
            
    utils.predict_results(y_true, preds,
                          target_ds,
                          record_id = record_id,
                          output_name=output
                          )


parser = argparse.ArgumentParser(description='add args for training the VarIPred model')
parser.add_argument('--df_path', '-p', default='../example/dataset', type=str)
parser.add_argument('--train_ds', '-tr', default='train', type=str)
parser.add_argument('--test_ds', '-ts', default='test', type=str)
parser.add_argument('--pred', '-i', default='target', type=str)
parser.add_argument('--output', '-o', default='VarIPred_output', type=str)
parser.add_argument('--train', '-t', action="store_true")

args = parser.parse_args()

if __name__ == '__main__':

    print('=============== Loading data... ===============')
    
    storage_path = args.df_path
    
    print(f'=============== Start getting embeddings ... ===============')

    # if necessary, update the parameters of the last MLP layer. 
    if args.train:
        train_df = pd.read_csv(f'{storage_path}/{args.train_ds}.csv')
        test_df = pd.read_csv(f'{storage_path}/{args.test_ds}.csv')

        print(f'getting embeds for {args.train_ds}.csv')
        get_embeds(train_df, dataset = args.train_ds)
        
        print(f'getting embeds for {args.test_ds}.csv')
        get_embeds(test_df, dataset = args.test_ds)

        train_VarIPred(args.train_ds, args.test_ds)

    else:
        # predict the target df with VarIPred
        target_df = pd.read_csv(f'{storage_path}/{args.pred}.csv')
        get_embeds(target_df, dataset = args.pred)
        run_VarIPred(target_ds=args.pred, output=output)

    
    
    print('\n\n\n\n')
    print('=============== No Bug No Error, Finished!!! ===============')
    os.system('tput bel')
    
    print('\n\n\n\n')
