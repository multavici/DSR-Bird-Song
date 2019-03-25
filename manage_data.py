from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')

#dbm.clean_db()

dbm.make_selection(100, 1000)
dbm.download_missing()


df = dbm.selection_df()

df.groupby('label').count().min()

t, v = dbm.train_validation_split()

v.path = v.path.apply(lambda x: x.replace('.pkl', '.png'))

t.to_csv('top100_img_train.csv')
v.to_csv('top100_img_val.csv')
