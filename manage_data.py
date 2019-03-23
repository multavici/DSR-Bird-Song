from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')

dbm.make_selection(100, 300)
dbm.download_missing()

t, v = dbm.train_validation_split()


t.to_csv()
v.to_csv()
