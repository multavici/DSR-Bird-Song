from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')

dbm.make_selection(100, 1000)
dbm.download_missing()
