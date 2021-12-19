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
from transformers import AdamW, get_cosine_schedule_with_warmup, BertForSequenceClassification, BertConfig

from dataset import ABSABertDataset

np.set_printoptions(precision=5)


def train_loop(model, dataloader, optimizer, device, dataset_len):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits,dim=1)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / dataset_len

    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for batch in tqdm(dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits,dim=1)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy


def main(config):
    pprint(config)

    batch_size = config['batch_size']

    epochs = config['epochs']

    learning_rate = config['learning_rate']

    number_of_runs = config['num_runs']

    data_dir = config['data_dir']

    for i in range(number_of_runs):
        train_df = pd.read_csv(data_dir+"/absa_train.csv")
        cv_df = pd.read_csv(data_dir+"/absa_cv.csv")
        test_df = pd.read_csv(data_dir+"/absa_test.csv")

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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(device)

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
            loss, accuracy = train_loop(model,
                                        train_dataloader,
                                        optimizer,
                                        device,
                                        len(train_dataset))
            
            print("Train Loss: ", loss)
            print("Train Accuracy: ", accuracy)

            if scheduler is not None:
                scheduler.step()

            if loss >= best_loss:
                early_stop_counter += 1
            else:
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                best_loss = loss

            if early_stop_counter == early_stop_limit:
                break

        model.load_state_dict(best_model_wts)
        cv_loss, cv_accuracy= eval_loop(model,
                                    cv_dataloader,
                                    device,
                                    len(cv_dataset))
            
        print("Validation Loss: ", cv_loss)
        print("Validation Accuracy: ", cv_accuracy)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    args = parser.parse_args()
    main(args.__dict__)

