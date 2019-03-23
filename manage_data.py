from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')

dbm.make_selection(100, 300)
dbm.download_missing()

selec = dbm.selection_df()


selec
import matplotlib.pyplot as plt
selec.groupby('label').count().plot(kind='barh')
