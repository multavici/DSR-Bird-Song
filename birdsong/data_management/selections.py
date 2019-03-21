from .utils import sql_selectors

class Selection:
    """"""
    def __init__(self, db_conn, based_on='seconds', amount=6000):
        self.conn = db_conn
        assert based_on in ['seconds', 'classes'], "Selection can either be made based on desired nr. of 'classes' or minimum 'seconds' per class."
        self.based_on = based_on
        self.amount = amount

    def __call__(self):
        df = sql_selectors.lookup_all_recordings(self.conn)
        distribution = df.groupby('label').scraped_duration.sum().sort_values()
        
        if self.based_on == 'seconds':
            selection = distribution[distribution >= self.amount]
            print(f'There are {len(selection)} classes available with at least {self.amount} seconds.')
        
        if self.based_on == 'classes':
            selection = distribution.tail(self.amount)
            print(f'There are at least {selection.min():.0f} seconds per class available for {self.amount} classes.')
        recordings_in_selection = df[df.label.isin(selection.index.values)]





import sqlite3
from birdsong.data_management.utils import sql_selectors

conn = sqlite3.connect('storage/db.sqlite')

sel = Selection(conn, based_on='classes', amount=100)

sel()









df = sql_selectors.lookup_all_recordings(conn)

distribution = df.groupby('label').scraped_duration.sum().sort_values()
selection = distribution[distribution >= 10000]

distribution.tail(300)
df.head()


selection

selection.index.values

test = df[df.label.isin(selection.index.values)]

test
