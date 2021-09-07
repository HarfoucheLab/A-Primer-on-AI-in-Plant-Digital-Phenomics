import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from datetime import datetime

## DISTRIBUTED:
import torch.multiprocessing as mp
import torch.distributed as dist


#########################
### TRAINING FUNCTION ###
#########################

def train(gpu, args):

    ## PARALLEL
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    ###

    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                        prototype_activation_function, add_on_layers_type, experiment_run

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    
    if rank == 0: #only the main process should write
        makedir(model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'), rank=rank, gpu=gpu)
    img_dir = os.path.join(model_dir, 'img')
    
    if rank == 0:
        makedir(img_dir)
    
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    from settings import train_dir, test_dir, train_push_dir, num_workers, \
                        train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # all datasets
    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, sampler=train_sampler)

    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
    ]))

    train_push_sampler = torch.utils.data.distributed.DistributedSampler(train_push_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False) #, sampler=train_push_sampler)

    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, sampler=test_sampler)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda(gpu)
    #ppnet_multi = torch.nn.DataParallel(ppnet)
    #ppnet_multi = torch.nn.parallel.DistributedDataParallel(ppnet, device_ids=[gpu], find_unused_parameters=True)

    class_specific = True

    # define optimizer
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from settings import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, min_saving_accuracy


    epoch = 0
	
    #######################
    ### RESUME TRAINING ###
    #######################

    if args.resume != '': #and rank == 0:
        checkpoint = torch.load(args.resume)
        ppnet.load_state_dict(checkpoint['model_state_dict'])
        joint_lr_scheduler.load_state_dict(checkpoint['joint_lr_scheduler_state_dict'])
        joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
        warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])
        last_layer_optimizer.load_state_dict(checkpoint['last_layer_optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        
    #######################
    ppnet_multi = torch.nn.parallel.DistributedDataParallel(ppnet, device_ids=[gpu], find_unused_parameters=True)


    # train the model
    log('start training')
    import copy
    start = datetime.now()

    while epoch < num_train_epochs:

        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=min_saving_accuracy, rank=rank, log=log)

        if epoch >= push_start and epoch in push_epochs:
            if rank == 0:
                push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=preprocess_input_function, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log)

            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=min_saving_accuracy, rank=rank, log=log)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=min_saving_accuracy, rank=rank, log=log)

            save.save_general_checkpoint(model=ppnet, epoch=epoch, joint_lr_scheduler=joint_lr_scheduler,
             joint_optimizer=joint_optimizer, warm_optimizer=warm_optimizer,
             last_layer_optimizer=last_layer_optimizer, model_dir=model_dir,
             model_name=str(epoch) + '_' + str(i) + 'checkpoint', accu=accu, target_accu=min_saving_accuracy,
             rank=rank, log=log)

        
        epoch = epoch + 1

    
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    logclose()



#########################

#####################
### MAIN FUNCTION ###
#####################
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    parser.add_argument('-rs', '--resume', default='', help='Resume training from saved checkpoint.')

    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '127.0.0.1'                 #'IPADDR' ##INSERT YOUR IP.
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    #print(os.environ['CUDA_VISIBLE_DEVICES'])
    

if __name__ == '__main__':
    main()
