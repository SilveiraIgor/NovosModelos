import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, mean_absolute_error
import csv
import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightning as pl
from lightning.pytorch.trainer import Trainer
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.csv_logs import CSVLogger
from IPython.display import display
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

data_path = "./"
loaded_data = load_from_disk(data_path)
batch_size = 1
epochs = 100
COMPETENCIA = 2

def discrepancia_horizontal(gold, pred):
    ocorrencias = 0
    total = 0
    for y, y_hat in zip(gold, pred):
        total += 1
        if abs(y - y_hat) > 80:
            ocorrencias += 1
    #print(f"Vai sair daqui com {ocorrencias} ocorrencias e {total} total: {ocorrencias/total}")
    return (ocorrencias/total)*100

def compute_metrics(preds, labels):
    labels = np.array(labels) *200
    preds = np.array(preds) * 200
    #print("Printar o labels e preds no Compute_metrics: ", len(labels), len(preds))
    mae = mean_absolute_error(labels, preds)
    #mae = 15
    RMSE = mean_squared_error(preds, labels, squared=False)
    MSE = mean_squared_error(preds, labels, squared=True)
    labels = discretizar(labels)
    preds = discretizar(preds)
    dv = discrepancia_horizontal(labels, preds)
    #print(f"Valor de dv: {dv}")
    QWK = cohen_kappa_score(preds, labels, labels=[0,40,80,120,160,200])
    acc = accuracy_score(labels, preds)
    return {"RMSE": RMSE, 'QWK': QWK,'MAE': mae, 'ACC': acc*100, 'MSE': MSE, 'HDIV': dv}

modelo = "neuralmind/bert-large-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo,model_max_length=512, truncation=True, do_lower_case=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.is_available()
from datasets import disable_caching

disable_caching()

def preparar_dataset(exemplo):
    resposta = tokenizer(exemplo['essay_text'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    return {'input_ids': resposta['input_ids'], 'token_type_ids': resposta['token_type_ids'], 'attention_mask':resposta['attention_mask']} 
    
def preparar2_dataset(exemplo):
    numero = exemplo['linked_items']['grades'][COMPETENCIA-1]
    return {'placeholder': numero/200}

def preparar_teste(exemplo):
    resposta = torch.DoubleTensor([exemplo['placeholder']])
    return {'labels': resposta}

def collate_batch(batch):
    input_ids, masks, types, labels = [], [], [], []
    for instancia in batch:
        input_ids.append(torch.tensor(instancia['input_ids']))
        masks.append(torch.tensor(instancia['attention_mask']))
        types.append(torch.tensor(instancia['token_type_ids']))
        nota = instancia['labels']
        labels.append(nota)
    input_ids = torch.cat(input_ids)
    masks = torch.cat(masks)
    types = torch.cat(types)
    labels = torch.FloatTensor(labels)
    return {'input_ids':input_ids, 'labels':labels, 'token_type_ids':types, 'attention_mask': masks}


dataset = loaded_data.map(preparar_dataset)
dataset = dataset.map(preparar2_dataset)
dataset = dataset.map(preparar_teste)
dataset2 = {}
dataset2['train'] = DataLoader(dataset['train'], batch_size=batch_size,collate_fn=collate_batch, shuffle=True)
dataset2['validation'] = DataLoader(dataset['validation'], batch_size=batch_size,collate_fn=collate_batch, shuffle=True)
dataset2['test'] = DataLoader(dataset['test'], batch_size=batch_size,collate_fn=collate_batch, shuffle=True)

def contar_discrepancias(vetor):
    contagem = 0
    for i in range(2):
        for j in range(i+1,3):
            if abs(vetor[i] - vetor[j]) > 80:
                contagem += 1
    return contagem

def filtrar_dataset(dataset):
    facil = []
    dificil = []
    id_notas = {}
    for questao in dataset:
        if questao['row_id'] not in id_notas:
            id_notas[questao['row_id']]  = [questao['placeholder']*200]
        else:
            id_notas[questao['row_id']].append(questao['placeholder']*200)
    #criei o dicionario com as notas
    print("Como ficou cada redacao: ", id_notas)
    for chave in id_notas:
        if len(id_notas[chave]) != 3:
            print("Deu algum erro")
    #testei para ver se todas as entradas tem tamanho 3
    for chave in id_notas:
        dificuldade = contar_discrepancias(id_notas[chave])
        if dificuldade == 0:
            facil.append(chave)
        elif dificuldade == 1 or dificuldade == 2:
            dificil.append(chave)
        else:
            print(">>>>>>>>Erro!!!<<<<<<<<")
    dataset_facil, dataset_dificil = [], []
    print(f"Tamanho dos splits: fácil = {len(facil)} e difícil = {len(dificil)}")
    for questao in dataset:
        if questao['row_id'] in facil:
            dataset_facil.append(questao)
        elif questao['row_id'] in dificil:
            dataset_dificil.append(questao)
        else:
            print("Erro na hora de quebrar o dataset")
    return {'facil': dataset_facil, 'dificil': dataset_dificil}
            

dataset_filtrado = filtrar_dataset(dataset['test']) 
dataset2['facil'] = DataLoader(dataset_filtrado['facil'], batch_size=batch_size,collate_fn=collate_batch)
dataset2['dificil'] = DataLoader(dataset_filtrado['dificil'], batch_size=batch_size,collate_fn=collate_batch)

def discretizar(vetor):
    v = []
    referencia = [0, 40, 80, 120, 160, 200]
    for elemento in vetor:
        novo_v = []
        for r in referencia:
            novo_v.append(abs(r - elemento))
        indice = np.argmin(novo_v)
        v.append(referencia[indice])
    return v

class TreinadorCustom(pl.LightningModule):
    def __init__(self, modelo, w_steps, t_steps):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=1).to(device)
        self.warmup_steps = w_steps
        self.num_training_steps = t_steps
        self.save_hyperparameters(ignore=['model'])
    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    def training_step(self, batch, batch_idx):
        y = batch['labels'].to(device)
        y = torch.flatten(y)
        y_hat = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)  , token_type_ids=batch['token_type_ids'].to(device) )
        y_hat = y_hat['logits']#.cpu().detach().numpy()
        y_hat = torch.flatten(y_hat)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss
    def validation_step(self, batch, batch_idx):
        """
        y = batch['labels'].to(device)
        y = torch.flatten(y)
        y_hat = self.model(input_ids=batch['input_ids'].to(device))
        y_hat = y_hat['logits']#.cpu().detach().numpy()
        y_hat = torch.flatten(y_hat)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        #print("Printando a loss da validacao: ", loss)
        retorno = compute_metrics(y, y_hat)
        #print(f"RMSE: {retorno['RMSE']}, QWK: {retorno['QWK']}")
        loss = 0.00
        return loss
        """
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def get_predictions_and_labels(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    all_true_labels = []
    i=0
    model.eval()
    for batch in tqdm(dataloader, desc="Obtaining predictions"):
        labels = batch["labels"].to(device)
        with torch.no_grad():
            output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)  , token_type_ids=batch['token_type_ids'].to(device))
            #predicted_classes = predict_classes(output) 
        # If using GPU, need to move the data back to CPU to use numpy.
        all_predictions.extend(output['logits'].cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

    return all_predictions, all_true_labels

from IPython.display import display
from lightning.pytorch.callbacks import Callback

class EpochEndCallback(Callback):
    def __init__(self):
        self.metrics_df = pd.DataFrame(columns=
                                       ['Epoch', 'Train Loss', 'Validation Loss',
                                         'Train QWK', 'Validation QWK',
                                         'Train RMSE', 'Validation RMSE', 
                                        'Train ACC', 'Validation ACC',
                                        'Train div', 'Validation div',
                                        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Metrics for training data
        train_dataloader = trainer.train_dataloader
        val_dataloader = trainer.val_dataloaders
        
        #epoch_train_loss_mean = torch.stack(pl_module.training_step_loss).mean()
        #epoch_val_loss_mean = torch.stack(pl_module.validation_step_loss).mean()
        #pl_module.training_step_loss.clear()
        #pl_module.validation_step_loss.clear()
        global rede2
        model = rede2
        train_predictions, train_true_labels = get_predictions_and_labels(model, train_dataloader)
        val_predictions, val_true_labels = get_predictions_and_labels(model, val_dataloader)
        retorno_val = compute_metrics(val_predictions, val_true_labels)
        retorno = compute_metrics(train_predictions, train_true_labels)
        train_qwk = retorno['QWK']
        val_qwk = retorno_val['QWK']
        val_rmse = retorno_val['RMSE']
        train_rmse = retorno['RMSE']
        pl_module.log('val_qwk', val_qwk)

        #print("Val: ", retorno_val)
        #print("train: ", retorno)
        new_row = {'Epoch': current_epoch, 
                    'Train Loss': retorno['MSE'],
                    'Validation Loss': retorno_val['MSE'],
                    'Train QWK': train_qwk,
                    'Validation QWK': val_qwk,
                    'Train RMSE': train_rmse,
                    'Validation RMSE': val_rmse,
                    'Train ACC': retorno['ACC'],
                    'Validation ACC': retorno_val['ACC'],
                   'Train div': retorno['HDIV'],
                   'Validation div': retorno_val['HDIV'],
                    }
        #print("New_row: ", new_row)
        new_row = pd.Series(new_row).to_frame().T
        self.metrics_df = pd.concat([self.metrics_df, new_row])
        display(self.metrics_df)

logger = CSVLogger("model_logs", name="enem_essay_score_regressor")
early_stop_callback = EarlyStopping(monitor="val_qwk", patience=3, verbose=True, mode="max")
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_qwk", mode="max")
num_training_steps = epochs * len(dataset2['train'])
warmup_steps = int(num_training_steps * 0.1)

rede2 = TreinadorCustom(modelo, warmup_steps, num_training_steps)
trainer = Trainer(max_epochs=100, callbacks=[EpochEndCallback(), early_stop_callback, checkpoint_callback ], logger=logger, accumulate_grad_batches=16)
trainer.fit(model=rede2, train_dataloaders=dataset2['train'], val_dataloaders=dataset2['validation'] )

display(checkpoint_callback.best_model_path)
display(checkpoint_callback.best_model_score)

best_model = TreinadorCustom.load_from_checkpoint(checkpoint_callback.best_model_path)

tipos = ['test', 'facil', 'dificil']
for t in tipos:
    all_predictions, all_true_labels = get_predictions_and_labels(best_model, dataset2[t])
    r = compute_metrics(all_true_labels, all_predictions)
    print(f"No {t}: ACC: {r['ACC']}, RMSE: {r['RMSE']}, QWK: {r['QWK']}, HDIV: {r['HDIV']} ") 