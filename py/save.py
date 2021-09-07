import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, rank=0, log=print):
    '''
    model: this is not the multigpu model
    '''
    if rank == 0:
        if accu > target_accu:
            log('\tabove {0:.2f}%'.format(target_accu * 100))
            # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
            torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def save_general_checkpoint(model, epoch, joint_lr_scheduler, joint_optimizer,
 warm_optimizer, last_layer_optimizer, model_dir, model_name, accu, target_accu, rank=0, log=print):
    if rank == 0:
        if accu > target_accu:
            log('saving checkpoint...')
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
                    'joint_optimizer_state_dict': joint_optimizer.state_dict(),
                    'warm_optimizer_state_dict': warm_optimizer.state_dict(),
                    'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
                    }, os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
            log('checkpoint saved!')
