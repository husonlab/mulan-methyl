import sys
import os
sys.path.append('/home/ubuntu/project/dna_methy/github/mulan-methyl/code')
os.chdir('/home/ubuntu/project/dna_methy/github/mulan-methyl/code')
#curPath = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.abspath(curPath))
import run_finetune
import data_processor
import run_evaluate
import compute_importance
import argparse
import torch


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, required=True, help='the file path of input data')
    # prediction
    parser.add_argument('--prediction', action='store_true', help='True if conduct prediction (after have finetuned models)')
    parser.add_argument('--species', default=None, required=False, help='species of input data, in full format if self define by user (s__Homo sapiens)')
    parser.add_argument('--methyl_type', default=None, required=True, help='methylation type, 6mA, 4mC or 5hmC')
    parser.add_argument('--developer', action='store_true', required=False, help='True for developer testing, the uploaded data has label, Presence means True')
    parser.add_argument('--output_file', default=None, required=False, help='the file path of generated result')
    parser.add_argument('--custom_length', action='store_true', required=False, help='True if length of sample is longer than 41')
    parser.add_argument('--do_visualize', action='store_true', required=False, help='True if conduct the computation of importance score')
    # data processing
    parser.add_argument('--data_proc', action='store_true', help='process DNA sequence to sentence')
    parser.add_argument('--data_type', default=None, required=False, help='the type(suffix) of input data, txt/tsv/fasta')
    parser.add_argument('--labelled', action='store_true', help='true if dataset has label')
    parser.add_argument('--custom_species', action='store_true', required=False, help='False if taxonomic lineage of data belongs to iDNA-MS, Presence means True')
    parser.add_argument('--data_output_dir', default=False, required=False, help='Directory of processed data')
    # fine-tuning
    parser.add_argument('--finetune', action='store_true', help='True if fine tuning models')
    parser.add_argument('--model_list', default=False, required=False, nargs='+', help='the model name for fine-tuning')
    parser.add_argument('--finetuned_output_dir', default=False, required=False, help='dir for saving fine-tuning result')
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    cur_path = os.getcwd()
    if args.prediction:
        func_data_processor = data_processor.DataProcesser()
        func_run_evaluate = run_evaluate.RunEvaluate()
        output_path = os.path.join(cur_path, args.output_file)
        processed_df = func_data_processor.data_loader(cur_path, args.input_file, args.data_type, args.species, args.methy_type, args.custom_species, args.developer, args.custom_length)
        prediction_df = func_run_evaluate.predict_func(args.species, processed_df, args.methy_type, cur_path, args.developer, args.custom_length)
        if not args.developer:
            if not args.custom_length:
                prediction_df.to_csv(output_path, sep='\t', index=False)
            if args.custom_length == True:
                func_compute_importance = compute_importance.ImportanceScore()
                prediction_pos_df = prediction_df[prediction_df['pred_label']==1][['id', 'seq']]
                processed_prediction_pos_df = func_data_processor.predicted_data_loader(cur_path, args.species, prediction_pos_df, args.custom_species, args.developer)
                importance_scores = func_compute_importance.myfunc(processed_prediction_pos_df, args.species, args.methy_type)
                prediction_pos_df['importance_score'] = list(importance_scores)
                prediction_pos_df['center_methylated'] = ''
                for index, row in prediction_pos_df.iterrows():
                    row['center_methylated'] = func_compute_importance.center_methylation_judgement(row)
                prediction_pos_df = prediction_pos_df[['id', 'seq', 'center_methylated']]
                prediction_pos_df.to_csv(output_path, sep='\t', index=False)
    elif args.finetune:
        func = run_finetune.Finetune()
        func._run_finetune(args.input_file, args.model_list, args.finetuned_output_dir, args.methyl_type)
    elif args.data_proc:
        func_ = data_processor.DataProcesser()
        processed_df = func_.data_loader(args.input_file, args.data_type, args.custom_species, args.labelled)
        processed_df.to_csv(f'{args.data_output_dir}/processed.tsv', sep='\t', index=False)





if __name__ == '__main__':
    main()
