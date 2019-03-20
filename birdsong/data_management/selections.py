from .utils import sql_selectors

class Selection:
    def __init__(self, db_conn, nr_classes, seconds_per_class):
        self.conn = db_conn
        self.nr_classes = nr_classes
        self.seconds_per_class = seconds_per_class
