
def upsample_df(df, samples_per_class):
    out = df.groupby('label').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
    return out.sample(frac=1).reset_index(drop=True)
