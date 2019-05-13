from birdsong.data_management.management import DatabaseManager

# Iniate storage directory
dbm = DatabaseManager('storage')

# Top 100 species, aim for 1800 spectrogram slices each
dbm.make_selection(100, 1800)

# Cleaning residual files and wrong db entries (from previous donwloads)
dbm.clean_storage()
dbm.clean_db()

# Prepare selection of background noise slices for augmentation
dbm.make_some_noise()

# Download and prepare recordings to make selection available
dbm.download_missing()

# Create training/validation split and store as .csv
t, v, c = dbm.train_validation_split()

t.to_csv('storage/dev_train.csv')
v.to_csv('storage/dev_val.csv')
c.to_csv('storage/dev_codes.csv')
