import argparse
from datetime import datetime
import dgl
import numpy as np
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import sys

from model import GAT
from utils import DataBatcher

def train():
    print("Starting training...")
    epochs = 1_000_000
    for epoch in range(epochs):
        model.train() 
        batch_size = 30       
        g, labels, f = data_batcher.get_batch(batch_size = batch_size, train = True)
        
        graphs = dgl.batch(g)    

        logits, attention = model(graphs, graphs.srcdata['x'])  

        labels = np.squeeze(labels)
        labels = torch.tensor(labels).long()
        loss = loss_func(logits, labels)

        writer.add_scalar('train/loss', loss, epoch)
        
        predicted_labels = torch.argmax(logits, 1)
        accuracy = ((labels.eq(predicted_labels.float())).sum()).float()/batch_size
        writer.add_scalar('train/accuracy', accuracy, epoch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if epoch % 20 == 0 : 

            labels = torch.squeeze(torch.Tensor(np.array(labels)).long())

            accuracy_func = BinaryAccuracy()
            accuracy = accuracy_func(predicted_labels, labels)
            writer.add_scalar('train_metrics/accuracy', accuracy, epoch)

            precision_func = BinaryPrecision()
            precision = precision_func(predicted_labels,labels)
            writer.add_scalar('train_metrics/precision', precision, epoch)

            recall_func = BinaryRecall()
            recall = recall_func(predicted_labels, labels)
            writer.add_scalar('train_metrics/recall', recall, epoch)

            f1 = 2*(precision*recall)/(precision+recall)
            writer.add_scalar('train_metrics/f1', f1, epoch)

            print_summary("Training", epoch, accuracy, precision, recall, f1)
        
        #evaluate on test set
        if epoch % 20 == 0 :
            model.eval()           
            g, labels, f = data_batcher.get_batch(batch_size = batch_size, train = False)

            graphs = dgl.batch(g)    
            logits, attention = model(graphs, graphs.srcdata['x'])

            predicted_labels = torch.argmax(logits, dim = 1)

            labels = torch.squeeze(torch.Tensor(np.array(labels)).long())

            accuracy_func = BinaryAccuracy()
            accuracy = accuracy_func(predicted_labels, labels)
            writer.add_scalar('metrics/acc', accuracy, epoch)

            precision_func = BinaryPrecision()
            pre = precision_func(predicted_labels,labels)
            writer.add_scalar('metrics/precision', precision, epoch)

            recall_func = BinaryRecall()
            rec = recall_func(predicted_labels, labels)
            writer.add_scalar('metrics/recall', recall, epoch)

            f1 = 2*(pre*rec)/(pre+rec)
            writer.add_scalar('metrics/f1', f1, epoch)
            print_summary("Test", epoch, accuracy, precision, recall, f1)
            model.train()
    
    print("Training complete!")

def print_summary(mode, epoch, accuracy, precision, recall, f1):
    print(f"{mode}: Epoch #{epoch} ({datetime.now()}); Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 = {f1}")

if __name__ == "__main__":
    # Parser code created with help from ChatGPT, 18 June 2024
    parser = argparse.ArgumentParser(description='Trains the model.')
    parser.add_argument(
        "--recalc",
        action="store_true",
        help="Recalculate the node labels"
    )
    args = parser.parse_args()
    if args.recalc:
        data_batcher = DataBatcher(recalc_labels=True)
    else:
        data_batcher = DataBatcher(recalc_labels=False)
    # End

    num_features = data_batcher.get_len_feature_space()
    model = GAT(num_features)
    optimiser = Adam(model.parameters(), lr=0.001) #TODO rate to high
    loss_func = CrossEntropyLoss()
    ct = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    writer = SummaryWriter('./runs/' + ct)
    train()