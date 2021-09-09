
from ..settings import *
def test_settings_created():
    assert 'base_architecture' in locals()
    assert 'img_size' in locals()
    assert 'prototype_shape' in locals()
    assert 'num_classes' in locals()
    assert 'prototype_activation_function' in locals()
    assert 'add_on_layers_type' in locals()
    assert 'experiment_run' in locals()
    assert 'data_path' in locals()
    assert 'train_dir' in locals()
    assert 'train_push_dir' in locals()
    assert 'test_dir' in locals()
    assert 'train_batch_size' in locals()
    assert 'test_batch_size' in locals()
    assert 'train_push_batch_size' in locals()