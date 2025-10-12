import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='CGR: Confidence-Guided Reply for Buffer-Based Continual Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    return parser

    
class Cgr(ContinualModel):
    NAME = 'cgr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Cgr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
    
    def end_epoch(self, dataset, train_loader):
        self.epoch += 1            

    def observe(self, inputs, labels, not_aug_inputs, index_):
        
        real_batch_size = inputs.shape[0]
        
        # batch update
        batch_x, batch_y = inputs, labels
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_combine = batch_x
        batch_y_combine = batch_y
            
        self.opt.zero_grad()    
        
        if self.buffer.is_empty():
            logits = self.net(batch_x_combine)
            novel_loss = self.loss(logits, batch_y_combine)
            
        else:
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
        
            mem_x = mem_x.to(self.device)
            mem_y = mem_y.to(self.device)
            mem_x_combine = mem_x
            mem_y_combine = mem_y

            combined_inputs = torch.cat([mem_x_combine, batch_x_combine])
            combined_labels = torch.cat((mem_y_combine, batch_y_combine))

            combined_logits = self.net(combined_inputs)
            novel_loss = self.loss(combined_logits, combined_labels)
        
        novel_loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                             labels=labels[:real_batch_size])
        
        return novel_loss.item()
