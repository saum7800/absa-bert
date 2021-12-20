import argparse
import copy
import os
import pickle
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup, BertForSequenceClassification, BertConfig, T5ForConditionalGeneration, T5Tokenizer
from metrics import get_metrics
from dataset import ABSABertDataset, T5Dataset
import wandb

np.set_printoptions(precision=5)


def train_loop(model, dataloader, optimizer, device, dataset_len, model_type, tokenizer, epoch_num):
    model.train()

    running_loss = 0.0
    final_preds = []
    final_labels = []


    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        if model_type == "BERT":
            logits = outputs[1]
            preds = torch.argmax(logits,dim=1)
            final_preds.append(preds.cpu().detach().numpy())
            final_labels.append(labels.cpu().detach().numpy())

        elif model_type == "T5":
            model_outputs = model.generate(input_ids)
            preds = [tokenizer.decode(model_outputs[x], skip_special_tokens=True).lower() for x in range(len(labels))]
            preds_nums = []
            for pred in preds:
                if pred.find("positive") != -1:
                    preds_nums.append(2)
                elif pred.find("negative") != -1:
                    preds_nums.append(0)
                else:
                    preds_nums.append(1)

            labels_nums = batch['labels_nums']
            final_preds.append(preds_nums)
            final_labels.append(labels_nums)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        

    epoch_loss = running_loss / len(dataloader)
    metrics = get_metrics(final_preds, final_labels, "train")
    metrics['train_loss'] = epoch_loss
    metrics['epoch'] = epoch_num

    return epoch_loss, metrics


def eval_loop(model, dataloader, device, dataset_len, model_type, tokenizer, epoch_num):
    model.eval()

    running_loss = 0.0
    final_preds = []
    final_labels = []

    for batch in tqdm(dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        if model_type == "BERT":
            logits = outputs[1]
            preds = torch.argmax(logits,dim=1)
            final_preds.append(preds.cpu().detach().numpy())
            final_labels.append(labels.cpu().detach().numpy())

        elif model_type == "T5":
            model_outputs = model.generate(input_ids)
            preds = [tokenizer.decode(model_outputs[x], skip_special_tokens=True).lower() for x in range(len(labels))]
            preds_nums = []
            for pred in preds:
                if pred.find("positive") != -1:
                    preds_nums.append(2)
                elif pred.find("negative") != -1:
                    preds_nums.append(0)
                else:
                    preds_nums.append(1)

            labels_nums = batch['labels_nums']
            final_preds.append(preds_nums)
            final_labels.append(labels_nums)


        running_loss += loss.item()


    epoch_loss = running_loss / len(dataloader)
    if epoch_num == -1:
        metrics = get_metrics(final_preds, final_labels, "test")
        metrics['test_loss'] = epoch_loss
    else:
        metrics = get_metrics(final_preds, final_labels, "cv")
        metrics['cv_loss'] = epoch_loss

    return epoch_loss, metrics

def train_with_wandb(config):
    run = wandb.init(project="absa-enterpret", entity="saumb7800")
    wandb.config = config

    with wandb.init() as run:
        pprint(config)

        model_type = config['model_type']

        batch_size = config['batch_size']

        epochs = config['epochs']

        learning_rate = config['learning_rate']

        data_dir = config['data_dir']

        tokenizer = None

        
        train_df = pd.read_csv(data_dir+"/absa_train.csv")
        cv_df = pd.read_csv(data_dir+"/absa_cv.csv")
        test_df = pd.read_csv(data_dir+"/absa_test.csv")

        if model_type == 'BERT':
            train_dataset = ABSABertDataset(train_df['text'].tolist(), train_df['aspect'].tolist(), train_df['label'].tolist())
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            
            cv_dataset = ABSABertDataset(cv_df['text'].tolist(), cv_df['aspect'].tolist(), cv_df['label'].tolist())
            cv_dataloader = DataLoader(
                cv_dataset, batch_size=batch_size, shuffle=True)
            
            test_dataset = ABSABertDataset(test_df['text'].tolist(), test_df['aspect'].tolist(), test_df['label'].tolist())
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True)
                

            model_config = BertConfig.from_pretrained('bert-base-uncased')
            model_config.num_labels = 3
            model = BertForSequenceClassification(model_config)

        elif model_type == 'T5':
            train_dataset = T5Dataset(train_df['text'].tolist(), train_df['aspect'].tolist(), train_df['label'].tolist())
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            
            cv_dataset = T5Dataset(cv_df['text'].tolist(), cv_df['aspect'].tolist(), cv_df['label'].tolist())
            cv_dataloader = DataLoader(
                cv_dataset, batch_size=batch_size, shuffle=True)
            
            test_dataset = T5Dataset(test_df['text'].tolist(), test_df['aspect'].tolist(), test_df['label'].tolist())
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True)
                
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print("device: ", device)

        optimizer = AdamW(model.parameters(),
                        lr=learning_rate)

        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, num_warmup_steps=10, num_training_steps=epochs)

        scheduler = None

        early_stop_counter = 0
        early_stop_limit = config['early_stop']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for epoch_num in range(epochs):
            model.load_state_dict(best_model_wts)
            loss, metrics = train_loop(model,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset),
                                        model_type,
                                        tokenizer,
                                        epoch_num)
            
            print("Train Loss: ", loss)
            print("Train metrics: ")
            pprint(metrics)

            cv_loss, cv_metrics= eval_loop(model,
                                        cv_dataloader,
                                        device,
                                        len(cv_dataset),
                                        model_type,
                                        tokenizer,
                                        epoch_num)
                
            print("Validation Loss: ", cv_loss)
            print("Validation Metrics:")
            pprint(cv_metrics)

            wandb_logging_dict = {}

            for k, v in metrics.items():
                wandb_logging_dict[k] = v

            for k, v in cv_metrics.items():
                wandb_logging_dict[k] = v
            
            wandb.log(wandb_logging_dict)

            if scheduler is not None:
                scheduler.step()

            if cv_loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = cv_loss

            if early_stop_counter == early_stop_limit:
                break

        model.load_state_dict(best_model_wts)
        test_loss, test_metrics= eval_loop(model,
                                    test_dataloader,
                                    device,
                                    len(test_dataset),
                                    model_type,
                                    tokenizer,
                                    -1)
        
        print("Test Loss: ", test_loss)
        print("Test Metrics:")
        pprint(test_metrics)
        wandb.log(test_metrics)
        torch.save(model.state_dict(), '/content/drive/MyDrive/save_model_'+model_type+str(datetime.now()))

def main(config):
    pprint(config)

    model_type = config['model_type']

    batch_size = config['batch_size']

    epochs = config['epochs']

    learning_rate = config['learning_rate']

    number_of_runs = config['num_runs']

    data_dir = config['data_dir']

    tokenizer = None

    for i in range(number_of_runs):
        train_df = pd.read_csv(data_dir+"/absa_train.csv")
        cv_df = pd.read_csv(data_dir+"/absa_cv.csv")
        test_df = pd.read_csv(data_dir+"/absa_test.csv")

        if model_type == 'BERT':
            train_dataset = ABSABertDataset(train_df['text'].tolist(), train_df['aspect'].tolist(), train_df['label'].tolist())
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            
            cv_dataset = ABSABertDataset(cv_df['text'].tolist(), cv_df['aspect'].tolist(), cv_df['label'].tolist())
            cv_dataloader = DataLoader(
                cv_dataset, batch_size=batch_size, shuffle=True)
            
            test_dataset = ABSABertDataset(test_df['text'].tolist(), test_df['aspect'].tolist(), test_df['label'].tolist())
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True)
                

            model_config = BertConfig.from_pretrained('bert-base-uncased')
            model_config.num_labels = 3
            model = BertForSequenceClassification(model_config)

        elif model_type == 'T5':
            train_dataset = T5Dataset(train_df['text'].tolist(), train_df['aspect'].tolist(), train_df['label'].tolist())
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            
            cv_dataset = T5Dataset(cv_df['text'].tolist(), cv_df['aspect'].tolist(), cv_df['label'].tolist())
            cv_dataloader = DataLoader(
                cv_dataset, batch_size=batch_size, shuffle=True)
            
            test_dataset = T5Dataset(test_df['text'].tolist(), test_df['aspect'].tolist(), test_df['label'].tolist())
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=True)
                
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print("device: ", device)

        optimizer = AdamW(model.parameters(),
                          lr=learning_rate)

        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer, num_warmup_steps=10, num_training_steps=epochs)

        scheduler = None

        early_stop_counter = 0
        early_stop_limit = config['early_stop']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.inf

        for _ in range(epochs):
            model.load_state_dict(best_model_wts)
            loss, metrics = train_loop(model,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset),
                                        model_type,
                                        tokenizer)
            
            print("Train Loss: ", loss)
            print("Train metrics: ")
            pprint(metrics)

            cv_loss, cv_metrics= eval_loop(model,
                                        cv_dataloader,
                                        device,
                                        len(cv_dataset),
                                        model_type,
                                        tokenizer)
                
            print("Validation Loss: ", cv_loss)
            print("Validation Metrics:")
            pprint(cv_metrics)

            if scheduler is not None:
                scheduler.step()

            if cv_loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = cv_loss

            if early_stop_counter == early_stop_limit:
                break

        model.load_state_dict(best_model_wts)
        test_loss, test_metrics= eval_loop(model,
                                    test_dataloader,
                                    device,
                                    len(test_dataset),
                                    model_type,
                                    tokenizer)
            
        print("Test Loss: ", test_loss)
        print("Test Metrics:")
        pprint(test_metrics)
    
    torch.save(model.state_dict(), '/content/drive/MyDrive/save_model_'+model_type+str(datetime.now()))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model-type", type=str, default="BERT",
                        help="model type")

    parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size")

    parser.add_argument("--epochs", type=int, default=15,
                        help="number of epochs")

    parser.add_argument("--num-runs", type=int, default=1,
                        help="number of runs")

    parser.add_argument("--early-stop", type=int, default=10,
                        help="early stop limit")

    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="learning rate")

    parser.add_argument("--data-dir", type=str, default="",
                        help="directory for data")
    
    parser.add_argument("--save-dir", type=str, default="",
                    help="save model")

    args = parser.parse_args()
    # main(args.__dict__)
    train_with_wandb(args.__dict__)

