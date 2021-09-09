import sys
import os
import sys,os
sys.path.append(os.getcwd())
from settings import *
def test_settings_created():
    assert 'base_architecture' in globals()
    assert 'img_size' in globals()
    assert 'prototype_shape' in globals()
    assert 'num_classes' in globals()
    assert 'prototype_activation_function' in globals()
    assert 'add_on_layers_type' in globals()
    assert 'experiment_run' in globals()
    assert 'data_path' in globals()
    assert 'train_dir' in globals()
    assert 'train_push_dir' in globals()
    assert 'test_dir' in globals()
    assert 'train_batch_size' in globals()
    assert 'test_batch_size' in globals()
    assert 'train_push_batch_size' in globals()