from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')
dbm.make_selection(100, 1500)

dbm.make_some_noise()

dbm.clean_db()


dbm.download_missing()

av = dbm.selection_df()
av.groupby('label').count().sort_values(by='path')


t, v, codes = dbm.train_validation_split()

import pandas as pd
c = pd.DataFrame.from_dict(codes, orient='index').reset_index()
c= c.rename(columns={'index':'code', 0:'name'})
c

t.to_csv('top100_train.csv')
v.to_csv('top100_val.csv')
c.to_csv('top100_codes.csv')
