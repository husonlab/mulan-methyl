import pandas as pd

df = pd.read_csv('/home/ubuntu/project/dna_methy/github/mulan/data/benchmark/processed_dataset/test/test.tsv', sep='\t')
df.reset_index(drop=False, inplace=True)
df.rename(columns={'index':'id'}, inplace=True)
df['label'] = list(map(lambda x: x.split('_')[0], df['combine_label']))
df.loc[df.label==f'neg', 'label'] = 0
df.loc[df.label==f'pos', 'label'] = 1
df.rename(columns={'methy_type':'methyl_type'}, inplace=True)
df = df[['id', 'seq', 'species', 'methyl_type', 'label']]
df.to_csv('/home/ubuntu/project/dna_methy/github/mulan/data/benchmark/processed_dataset/test/test_set.tsv', sep='\t', index=False)

df = pd.read_csv('/home/ubuntu/project/dna_methy/github/mulan/data/benchmark/processed_dataset/train/processed_6mA.tsv', sep='\t')
df.reset_index(drop=False, inplace=True)
df.rename(columns={'index':'id'}, inplace=True)
df.to_csv('/home/ubuntu/project/dna_methy/github/mulan/data/benchmark/processed_dataset/train/processed_6mA.tsv', sep='\t', index=False)
