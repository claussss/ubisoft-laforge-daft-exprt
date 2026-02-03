from torch.utils.tensorboard import SummaryWriter


class DaftExprtLogger(SummaryWriter):
    def __init__(self, logdir):
        super(DaftExprtLogger, self).__init__(logdir)

    def log_training(self, loss, indiv_loss, grad_norm, learning_rate, duration, iteration):
        '''Log to TensorBoard: total train loss, individual train losses, grad norm.'''
        self.add_scalar('train/loss', loss, iteration)
        self.add_scalar('train/grad_norm', grad_norm, iteration)
        for key, value in indiv_loss.items():
            if value != 0:
                self.add_scalar(f'train/{key}', value, iteration)

    def log_validation(self, val_loss, val_indiv_loss, val_targets, val_outputs, model, hparams, iteration):
        '''Log to TensorBoard: total val loss and individual val losses only.'''
        self.add_scalar('val/loss', val_loss, iteration)
        for key, value in val_indiv_loss.items():
            self.add_scalar(f'val/{key}', value, iteration)
