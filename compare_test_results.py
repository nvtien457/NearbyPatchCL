import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = {
    '1%': '../checkpoints/Tien_2B-2Neg_SupCon0_20_256/finetune_e275_p1/test_result.csv',
    '5%': '../checkpoints/Tien_2B-2Neg_SupCon0_20_256/finetune_e275_p5/test_result.csv',
    '100%': '../checkpoints/Tien_2B-2Neg_SupCon0_20_256/finetune_e275_p100/test_result.csv'
}

total_df = []
for k, v in csv_path.items():
    df = pd.read_csv(v)
    # print(df.head(5))
    df['name'] = k
    df = df.melt(id_vars=['subset', 'name'],
                    var_name='metric',
                    value_name='value')
    total_df.append(df)

tdf = pd.concat(total_df, axis=0, ignore_index=True)
tdf.fillna(0, inplace=True)

tdf = tdf[~tdf['metric'].isin(['f1', 'balanced_acc'])]
groups = tdf.groupby('metric')
fig, axes = plt.subplots(len(groups), 1, sharex=True, sharey=True, figsize=(10, 6*len(groups)))

for (name, group), ax in zip(groups, axes):
    group.pivot_table(index='subset', columns='name', values='value').plot.barh(legend=True, ax=ax)
    ax.set_title(name)

fig.supylabel('value')
plt.tight_layout()
plt.savefig('t.jpg')