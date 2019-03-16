import pandas as pd

def upsample_df(df, samples_per_class):
    out = df.groupby('label').apply(lambda x: x.sample(n=samples_per_class, replace=True))
    return out.reset_index(drop=True)
