import data_processor
import run_evaluate
import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, required=True, help='the file path of input data')
    parser.add_argument('--data_type', default=None, required=True, help='the type(suffix) of input data, txt/tsv/fasta')
    parser.add_argument('--species', default=None, required=True, help='species of input data, in full format if self define by user (s__Homo sapiens)')
    parser.add_argument('--custom_species', action='store_true', required=False, help='if false then the taxonomic lineage of data belongs to iDNA-MS, Presence means True')
    parser.add_argument('--methy_type', default=None, required=True, help='methylation type, 6mA, 4mC or 5hmC')
    parser.add_argument('--developer', action='store_true', required=False, help='True for developer testing, the uploaded data has label, Presence means True')
    parser.add_argument('--output_file', default=None, required=None, help='the file path of generated result')
    args = parser.parse_args()
    return args


def func_predict():
    args = create_parser()
    cur_path = os.getcwd()
    func_data_processor = data_processor.DataProcesser()
    func_run_evaluate = run_evaluate.RunEvaluate()
    output_path = os.path.join(cur_path, args.output_file)
    processed_df = func_data_processor.data_loader(cur_path, args.input_file, args.data_type, args.species, args.custom_species, args.developer)
    prediction_df = func_run_evaluate.predict_func(args.species, processed_df, args.methy_type, cur_path, args.developer)
    if not args.developer:
        prediction_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    func_predict()
