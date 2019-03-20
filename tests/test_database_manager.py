from birdsong.data_management.management import DatabaseManager
from birdsong.datasets.tools.io import slice_audio
import pytest
import os
import shutil
import numpy as np

@pytest.fixture(scope='session')
def test_dir(tmpdir_factory):
    dir_ = tmpdir_factory.mktemp("storage")
    shutil.copy('storage/db.sqlite', dir_.join('db.sqlite'))
    return dir_

def test_slicing():
    sr = 22050
    audio = np.random.randn(int(5 *sr))
    slices = slice_audio(audio, sr, 5000, 2500)
    assert len(slices) == 1
    assert len(slices[0]) == 5 * sr

def test_slicer(test_dir):
    dbm = DatabaseManager(test_dir)
    url = 'https://www.xeno-canto.org/45555/download'
    dbm.SignalSlicer([('7', url)])
    assert '7_0.pkl' in os.listdir(dbm.signal_dir)

# Depends on previous test
def test_inventory(test_dir):
    dbm = DatabaseManager(test_dir)
    dbm.inventory()
    assert dbm.counts == {'7': 1}

# Depends on previous test
def test_df_creation(test_dir):
    dbm = DatabaseManager(test_dir)
    df = dbm.get_df()
    assert df.path[0] == '7_0.pkl'
    assert df.label[0] == 'rhea_americana'

"""
def test_download_below_median(test_dir):
    dbm = DatabaseManager(test_dir)
    diff = dbm.download_below_median(max_classes = 2, max_recordings = 1)
    print(diff)
"""
