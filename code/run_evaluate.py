import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizerFast, PreTrainedTokenizerFast, AutoTokenizer,BertTokenizer, DistilBertTokenizer, BertForMaskedLM, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer, ElectraTokenizer
from transformers import ElectraForSequenceClassification, XLNetForSequenceClassification, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, average_precision_score
import multiprocessing
from transformers import DataCollatorForLanguageModeling
from collections import Counter
from copy import deepcopy
import logging
import datetime

logger = logging.getLogger(__name__)
#torch.cuda.set_per_process_memory_fraction(0.8)


# Dataset generation
class MethyDataset(Dataset):
    def __init__(self, index, encodings, labels, labelled=False):
        self.index = index
        self.encodings = encodings
        self.labels = labels
        self.developer = labelled

    # read sample
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.developer:
            item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        if self.developer:
            return len(self.labels)
        else:
            return len(self.index)



class RunEvaluate(object):

    def evaluation_metrics(self, preds, probs, labels):
        f1 = f1_score(y_true=labels, y_pred=preds)
        acc = accuracy_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(labels, preds)
        roc_auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
        return {"accuracy": acc, "f1": f1, "recall":recall, "precision":precision, "mcc":mcc, "auc":roc_auc, "aupr": aupr}


    def predict_func(self, processed_df, methy_type, output_dir, finetuned_model_dir, labelled=False, multi_species=False, data_processed=False, species=None):
        '''
        :param species: species name
        :param processed_df: processed dataset ['id', 'seq', 'text']
        :param methy_site:
        :param output_dir: the path of the project
        :return:
        '''
        # build saving path
        log_path = os.path.join(output_dir, 'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # for multi species dataset
        if multi_species:
            processed_df = processed_df[processed_df['species']==species]
        if labelled:
            date = str(datetime.date.today())
            with open(f'{log_path}/pred_record.txt', 'a') as writer:
                writer.write(f'**** {date}_{methy_type} jointly predict on {species} ****' + '\n')
        x_test = processed_df['text'].tolist()
        if labelled:
            test_label = processed_df['label'].tolist()
        else:
            test_label = None
        test_index = processed_df['id'].tolist()
        model_list = ['XLNet', 'BERT', 'DistilBERT', 'ALBERT', 'ELECTRA']
        model_config = {
            'DistilBERT': (DistilBertTokenizer, DistilBertForSequenceClassification),
            'ALBERT': (AutoTokenizer, AlbertForSequenceClassification),
            'XLNet': (AutoTokenizer, XLNetForSequenceClassification),
            'BERT': (BertTokenizer, BertForSequenceClassification),
            'ELECTRA': (ElectraTokenizer, ElectraForSequenceClassification)
        }
        for model_ in model_list:
            tokenizer_type, model_type = model_config[model_]
            # load tokenizer
            tokenizer = tokenizer_type.from_pretrained(f'wenhuan/MuLan-Methyl-{model_}')
            # tokenizer
            if model_ in ['ALBERT', 'XLNet']:
                test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=200, return_tensors='pt')
            else:
                test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=100, return_tensors='pt')
            testDataset = MethyDataset(test_index, test_encoding, test_label, labelled)
            # predict on test set
            model = model_type.from_pretrained(f'{finetuned_model_dir}/{model_}/{methy_type}/model', num_labels=2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if model_ == 'XLNet':
                try:
                    args = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=64)
                    trainer = Trainer(args=args, model=model)
                    pred = trainer.predict(testDataset)
                except Exception:
                    try:
                        args = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=2)
                        trainer = Trainer(args=args, model=model)
                        pred = trainer.predict(testDataset)
                    except Exception:
                        args = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=1)
                        trainer = Trainer(args=args, model=model)
                        pred = trainer.predict(testDataset)
            else:
                args = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=64)
                trainer = Trainer(args=args, model=model)
                pred = trainer.predict(testDataset)
            if labelled:
                labels = list(pred.label_ids)
            probs_ = torch.as_tensor(pred.predictions)
            probs = torch.softmax(probs_,1).numpy()
            preds = np.argmax(probs, axis=1)
            if labelled:
                # evaluate model and save result
                result = self.evaluation_metrics(preds, probs[:,1], labels)
                res = str(model_) + ' '
                with open(f'{log_path}/pred_record.txt', 'a') as writer:
                    if model_ == 'XLNet':
                        header = 'model '
                        for key in sorted(result.keys()):
                            header = header + str(key) + ' '
                        writer.write(header + '\n')
                    for key in sorted(result.keys()):
                        res = res + str(result[key])[:7] + ' '
                    writer.write(res + '\n')
            if model_ == 'XLNet':
                all_probs = deepcopy(probs)
                cat_probs = deepcopy(probs)
            else:
                all_probs += probs
                cat_probs = np.concatenate((cat_probs, probs), axis=1)
        all_probs = all_probs/len(model_list)
        all_preds = np.argmax(all_probs, axis=1)
        if labelled:
            labels_ = labels
            labels = np.array(labels)
            labels = labels.reshape(labels.shape[0], 1)
            ensemble_results = self.evaluation_metrics(all_preds, all_probs[:,1], labels)
            jointRes = 'joint '
            with open(f'{log_path}/pred_record.txt', 'a') as writer:
                logger.info("***** Ensemble results *****")
                for key in sorted(ensemble_results.keys()):
                    logger.info("  %s = %s", key, str(ensemble_results[key]))
                    jointRes = jointRes + str(ensemble_results[key])[:7] + ' '
                writer.write(jointRes + '\n')
        MuLan_probs_df = pd.DataFrame({'prob_0': all_probs[:,0], 'prob_1': all_probs[:,1], 'pred_label': all_preds})
        if not data_processed:
            if labelled:
                processed_df = pd.concat([processed_df[['id', 'seq', 'label']], MuLan_probs_df], axis=1)
            else:
                processed_df = pd.concat([processed_df[['id', 'seq']], MuLan_probs_df], axis=1)
        else:
            if labelled:
                processed_df = pd.concat([processed_df[['id', 'text', 'label']], MuLan_probs_df], axis=1)
            else:
                processed_df = pd.concat([processed_df[['id', 'text']], MuLan_probs_df], axis=1)
        return processed_df




