import os 
import numpy as np 
import torch

class Agent:
    def __init__(self, net, args, config) -> None:
        self.net = net
        self.args = args
        self.config = config
        self.device = args.device 
        self.last_action = -1
        
    def forward(self, hist_buffer):
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(hist_buffer)
            return y_pred
    
    def mask_action_space(self, y_pred):
        if self.last_action == 0 or self.last_action == 1 or self.last_action == 2:
            y_pred[0][0:3] = torch.min(y_pred)
        elif self.last_action == 3 or self.last_action == 4 or self.last_action == 5:
            y_pred[0][3:6] = torch.min(y_pred)
        elif self.last_action == 6 or self.last_action == 7 or self.last_action == 8:
            y_pred[0][6:9] = torch.min(y_pred)
        elif self.last_action > 8 and self.last_action != self.args.n_action - 1:
            y_pred[0][9:self.args.n_action - 1] = torch.min(y_pred)
        return y_pred
        
    def act(self, hist_buffer, eps, mode='train'):
        if mode == 'train' and eps < self.config.RANDOM_ACTION:
            return np.random.randint(0, self.args.n_action), 0
        else:
            y_pred = self.forward(hist_buffer)
            y_pred = self.mask_action_space(y_pred)
            act_res = torch.argmax(y_pred).item()
            self.last_action = act_res
            return act_res, y_pred[0][act_res].item()