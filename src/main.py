import os 
import torch
import numpy as np 
import deepgate as dg 
import copy
import gym
from collections import deque
import shutil

from config import get_parse_args
from rl.config import RL_Config
from rl.qnet import Q_Net
from rl.agent import Agent
from rl.trainer import Trainer
from rl.buffer import ReplayBuffer
from rl.env import solve_Env

if __name__ == '__main__':
    args = get_parse_args()
    config = RL_Config(args)
    print('==> Using settings {}'.format(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training in single device: ', args.device)
    
    # Create RL environments 
    net = Q_Net(args)
    buffer = ReplayBuffer(args, config)
    if args.resume:
        model_last_path = os.path.join(args.save_dir, 'qnet_last.pth')
        net.load(model_last_path)
        print('==> Load model from {}'.format(model_last_path))
        buffer_last_path = os.path.join(args.save_dir, 'buffer.pth')
        if os.path.exists(buffer_last_path):
            buffer.load(buffer_last_path)
            print('==> Load buffer from {}'.format(buffer_last_path))
        else:
            print('==> No buffer found, start from scratch')
    target_net = copy.deepcopy(net)
    rl_env = solve_Env(args=args, config=config)
    agent = Agent(net, args, config)
    trainer = Trainer(args, config, net, target_net, buffer)
    
    # Train 
    for train_times in range(args.train_times):
        for reset_times in range(config.RESET_TIMES):
            if reset_times == 0:
                rl_env.next_instance()
                eps = 0
                if len(rl_env.init_graph['edge_index']) == 0 or len(rl_env.init_graph['x']) == 0:
                    break
            obs = rl_env.reset()
            done = False
            tot_reward = 0
            print('==> Training: {:} / {:}, Problem: {}'.format(reset_times, config.RESET_TIMES, rl_env.problem_name))
        
            while not done:
                action, _ = agent.act(obs, eps, mode='train')
                next_obs, reward, done, info = rl_env.step(action)
                eps += 1
                buffer.add_transition(obs, action, reward, done)
                obs = next_obs
                tot_reward += reward
                
                if buffer.ctr >= config.OBSERVE:
                    train_info = trainer.step()
                    print('==> Step: {:}, Action: {}, Loss: {:.4f}, Average Q: {:.4f}'.format(trainer.step_ctr, action, train_info['loss'], train_info['average_q']))
                    if trainer.step_ctr % args.save_epoch == 0:
                        buffer_path = os.path.join(args.save_dir, 'buffer.pth')
                        buffer.save(buffer_path)
                        print('==> Save buffer to {}'.format(buffer_path))
                        model_last_path = os.path.join(args.save_dir, 'qnet_last.pth')
                        net.save(model_last_path)
                        model_path = os.path.join(args.save_dir, 'qnet_{:}.pth'.format(trainer.step_ctr))
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        shutil.copy(model_last_path, model_path)
                        print('==> Save model to {}'.format(model_path))
            
            trainer.update_target()
    