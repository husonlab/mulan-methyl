import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(curPath))
import run_finetune
import data_processor
import run_evaluate
import argparse
import torch
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, required=True, help='the file path of input data')
    parser.add_argument('--labelled', action='store_true', help='true if dataset has label')
    # data processing
    parser.add_argument('--data_proc', action='store_true', help='process DNA sequence to sentence')
    parser.add_argument('--data_type', default=None, required=False, help='the type(suffix) of input data, txt/tsv/fasta')
    parser.add_argument('--custom_species', action='store_true', required=False, help='False if taxonomic lineage of data belongs to iDNA-MS, Presence means True')
    parser.add_argument('--data_output_dir', default=False, required=False, help='Directory of processed data')
    # fine-tuning
    parser.add_argument('--finetune', action='store_true', help='True if fine tuning models')
    parser.add_argument('--model_list', default=False, required=False, nargs='+', help='the model name for fine-tuning')
    parser.add_argument('--finetuned_output_dir', default=False, required=False, help='dir for saving fine-tuning result')
    parser.add_argument('--learning_rate', default=False, required=False, nargs='+', type=float, help='learning rate of corresponding model')
    # prediction
    parser.add_argument('--prediction', action='store_true', help='True if conduct prediction (after have finetuned models)')
    parser.add_argument('--multi_species', action='store_true', help='True if input dataset contains not only one species, then dataset should have species columns. If not then species column is not mandatory.')
    parser.add_argument('--data_processed', action='store_true', help='True if input dataset already processed')
    parser.add_argument('--species', default=None, required=False, help='For dataset has multi species species, assign a specific species if you want to have the prediction species-wise. In full format if self define by user (s__Homo sapiens)')
    parser.add_argument('--methyl_type', default=None, required=True, help='methylation type, 6mA, 4mC or 5hmC')
    parser.add_argument('--prediction_output_dir', default=None, required=False, help='the directory of generated result')
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    if args.finetune:
        func = run_finetune.Finetune()
        func._run_finetune(args.input_file, args.model_list, args.finetuned_output_dir, args.methyl_type, args.learning_rate)
    elif args.data_proc:
        func_ = data_processor.DataProcesser()
        processed_df = func_.data_loader(args.input_file, args.data_type, args.custom_species, args.labelled)
        processed_df.to_csv(f'{args.data_output_dir}/processed.tsv', sep='\t', index=False)
    elif args.prediction:
        func_data_processor = data_processor.DataProcesser()
        func_run_evaluate = run_evaluate.RunEvaluate()
        if args.data_processed:
            processed_df = func_data_processor.data_loader_processed(args.input_file, args.custom_species)
        elif not args.data_processed and args.multi_species:
            processed_df = func_data_processor.data_loader(args.input_file, args.data_type, args.custom_species, args.labelled)
        elif not args.data_processed and not args.multi_species:
            processed_df = func_data_processor.data_loader_predict(args.input_file, args.data_type, args.species, args.custom_species, args.labelled)
        prediction_df = func_run_evaluate.predict_func(processed_df, args.methyl_type, args.prediction_output_dir, args.finetuned_output_dir, args.labelled, args.multi_species, args.data_processed, args.species)
        prediction_df.to_csv(os.path.join(args.prediction_output_dir, f'{args.species}_res.csv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
