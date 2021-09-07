base_architecture = 'densenet161'
img_size = 224
prototype_shape = (2000, 128, 1, 1)
num_classes = 5
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '006_cassava_resume300'

data_path = 'PATH_TO_DATASET'
train_dir = data_path + 'PATH_TO_AUGMENTED_TRAINING_FOLDER'
train_push_dir = data_path + 'PATH_TO_TRAINING_FOLDER/'
test_dir = data_path + 'PATH_TO_VALIDATION_FOLDER'
train_batch_size = 40
test_batch_size = 40
train_push_batch_size = 64

num_workers=3
min_saving_accuracy=0.05

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
