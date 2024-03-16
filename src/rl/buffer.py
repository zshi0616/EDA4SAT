import torch 
import numpy as np 

class ReplayBuffer: 
    def __init__(self, args, config):
        self.ctr = 0        # Length
        self.full = False
        self.size = config.REPLAY_MEMORY
        self.device = args.device
        self.dones = torch.ones(self.size)
        self.rewards = torch.zeros(self.size)
        self.actions = torch.zeros(self.size, dtype=torch.long)
        # dtype=object allows to store references to objects of arbitrary size
        self.observations = []
        for idx in range(self.size):
            self.observations.append(None)

    def add_transition(self, obs, a, r_next, done_next):
        self.dones[self.ctr] = int(done_next)
        self.rewards[self.ctr] = r_next
        self.actions[self.ctr] = a

        # should be vertex_data, edge_data, connectivity, global
        self.observations[self.ctr] = obs

        if (self.ctr + 1) % self.size == 0:
            self.ctr = 0
            self.full = True
        else:
            self.ctr += 1

    def sample(self, batch_size):
        # to be able to grab the next, we use -1
        curr_size = self.ctr - 1 if not self.full else self.size - 1
        indexes = np.random.choice(range(0, curr_size), batch_size)
        batch_obs = []
        batch_next_obs = []
        for k in indexes:
            batch_obs.append(self.observations[k])
            batch_next_obs.append(self.observations[k + 1])
        batch_obs = torch.stack(batch_obs).squeeze(1)
        batch_next_obs = torch.stack(batch_next_obs).squeeze(1)
        
        return (
            batch_obs,
            self.actions[indexes].to(self.device),
            self.rewards[indexes].to(self.device),
            batch_next_obs,
            (1.0 - self.dones[indexes]).to(self.device),
        )
    
    def save(self, save_path):
        torch.save({
            'ctr': self.ctr,
            'full': self.full,
            'size': self.size,
            'dones': self.dones,
            'rewards': self.rewards,
            'actions': self.actions,
            'observations': self.observations
        }, save_path)
        
    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.ctr = checkpoint['ctr']
        self.full = checkpoint['full']
        self.size = checkpoint['size']
        self.dones = checkpoint['dones']
        self.rewards = checkpoint['rewards']
        self.actions = checkpoint['actions']
        self.observations = checkpoint['observations']
