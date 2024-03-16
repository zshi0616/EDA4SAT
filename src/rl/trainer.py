from torch import nn
from torch_scatter import scatter_max
import torch
from torch.optim.lr_scheduler import StepLR

class Trainer:
    def __init__(self, args, config, net, target, buffer):
        self.args = args
        self.config = config
        self.net = net
        self.target = target
        self.target.eval()
        self.buffer = buffer
        self.device = args.device
        self.step_ctr = 0
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=args.lr_step, gamma=0.5)
        self.loss = nn.MSELoss().to(self.device)
    
    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())
        
    def step(self):
        s, a, r, s_next, nonterminal = self.buffer.sample(self.config.BATCH_SIZE)
        
        # Target Q Net
        with torch.no_grad():
            target_qs = self.target(s_next)
            target_qs = torch.max(target_qs, dim=1).values
            targets = r.to(self.device) + nonterminal.to(self.device) * self.config.GAMMA * target_qs.to(self.device)
            targets = targets.to(self.device)

        # Q Net
        self.net.train()
        qs = self.net(s)
        qs = qs[torch.arange(len(a)), a]
        qs = qs.to(self.device)
        
        # Train
        loss = self.loss(qs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_ctr += 1
        
        # # Update target net
        # if self.step_ctr % self.config.UPDATE_TIME == 0:
        #     self.update_target()
        #     print('==> Update target net')
        
        return {
            'loss': loss.item(), 
            'lr': self.optimizer.param_groups[0]['lr'],
            'average_q': qs.mean().item()
        }