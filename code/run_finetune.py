import os
import pandas as pd
import torch
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer,BertTokenizer, DistilBertTokenizer, ElectraTokenizer, \
    BertForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification, XLNetForSequenceClassification, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# dataloader
class MethyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

class Finetune(object):

    def _run_finetune(self, dataPath, modelList, outputDir, methylType, learningRate=None):
        # read dataset
        mydf = pd.read_csv(dataPath, sep='\t')
        mydf['species'] = list(map(lambda x: x.split('.')[1].split(',')[1].lstrip('its species is'), mydf['text']))
        # split dataset
        x_train, x_test, train_label, test_label = train_test_split(mydf['text'], mydf['label'], test_size=0.2, stratify=mydf[['species', 'label']], random_state=22)
        x_train = x_train.tolist()
        x_test = x_test.tolist()
        train_label = train_label.tolist()
        test_label = test_label.tolist()
        #model_list = ['xlnet', 'bert', 'distilbert', 'albert', 'electra']
        model_config = {
            'DistilBERT': (DistilBertTokenizer, DistilBertForSequenceClassification),
            'ALBERT': (AutoTokenizer, AlbertForSequenceClassification),
            'XLNet': (AutoTokenizer, XLNetForSequenceClassification),
            'BERT': (BertTokenizer, BertForSequenceClassification),
            'ELECTRA': (ElectraTokenizer, ElectraForSequenceClassification)
        }
        if learning_rate is not None:
            lr_key = modelList
            lr_value = learningRate
        else:
            lr_key = ['BERT', 'DistilBERT', 'ALBERT', 'XLNet', 'ELECTRA']
            lr_value = [1e-5, 1e-5, 5e-5, 2e-5, 1e-5]
        lr_dict = dict(zip(lr_key, lr_value))
        # process for each model in the model list
        for model_ in modelList:
            pretrained_model_path = f'wenhuan/MuLan-Methyl-{model_}'
            tokenizer_type, model_type = model_config[model_]
            # load tokenizer
            tokenizer = tokenizer_type.from_pretrained(pretrained_model_path)
            # tokenization
            if model_ in ['XLNet', 'ALBERT']:
                MAX_SEQ_LEN = 200
            else:
                MAX_SEQ_LEN = 100
            train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=MAX_SEQ_LEN)
            test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=MAX_SEQ_LEN)
            train_dataset = MethyDataset(train_encoding, train_label)
            #print(train_dataset[0])
            #print(tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids']))
            test_dataset = MethyDataset(test_encoding, test_label)
            # load model
            model = model_type.from_pretrained(pretrained_model_path, num_labels=2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            # define saving dir
            checkpoint_path = f'{outputDir}/{model_}/{methylType}/checkpoint'
            log_path = f'{outputDir}/{model_}/{methylType}/log'
            model_path = f'{outputDir}/{model_}/{methylType}/model'
            for obj in [checkpoint_path, log_path, model_path]:
                if not os.path.exists(obj):
                    os.makedirs(obj)
            if model_ == 'ALBERT':
                BATCH_SIZE = 96
                WARMUP_STEPS = 1000
            else:
                BATCH_SIZE = 64
                WARMUP_STEPS = 100
            LEARNING_RATE = lr_dict[model_]
            # define evaluation metrics
            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                pred_prob = pred.predictions
                f1 = f1_score(labels, preds)
                acc = accuracy_score(labels, preds)
                recall = recall_score(labels, preds)
                precision = precision_score(labels, preds)
                roc_auc = roc_auc_score(labels, pred_prob[:,1])
                return {"accuracy": acc, "f1": f1, "recall":recall, "precision":precision, "auc":roc_auc}

            # define training argument
            training_args = TrainingArguments(
                output_dir=checkpoint_path,          # output directory
                num_train_epochs=32,              # total number of training epochs
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                warmup_steps=WARMUP_STEPS,
                weight_decay=0.01,               # strength of weight decay
                logging_dir=log_path,            # directory for storing logs
                do_predict=True,
                learning_rate=LEARNING_RATE,
                disable_tqdm=False,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                do_eval=True,
                load_best_model_at_end=True,
                metric_for_best_model='auc',
                logging_strategy='epoch',
                save_total_limit=1,
                seed=42,
                dataloader_drop_last=True,
                report_to='tensorboard'
            )
            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args, # training arguments, defined above
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,         # training dataset
                eval_dataset=test_dataset,             # evaluation dataset
                callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
                tokenizer=tokenizer
            )

            # record time
            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)
            print(f'start finetuning MuLan-Methyl-{model_} on {methylType} site')
            #start.record()
            trainer.train()
            trainer.evaluate()
            #end.record()

            trainer.save_model(model_path)
            #print(start.elapsed_time(end))
            print(f'MuLan-Methyl-{model_} on {methylType} site is finetuned')

