import sqlite3    
from sql_selectors import lookup_recordings_to_download
import pytest

@pytest.fixture
def get_cursor():
    conn = sqlite3.connect('storage/db.sqlite')
    c = conn.cursor()
    return c

# wont work anymore once more are downloaded and cygnus olor is depleted
def test_lookup_recordings_to_download(get_cursor):
    label = 'cygnus_olor'
    recordings = lookup_recordings_to_download(get_cursor, label, 5)
    assert len(recordings) == 5
