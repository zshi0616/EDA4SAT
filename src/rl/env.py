import numpy as np
import gym
import random
import os 
import sys
import glob
import shutil
import deepgate as dg
import torch
import time

from utils.utils import run_command
from utils.cnf_utils import kissat_solve, kissat_solve_dec
from utils.lut_utils import parse_bench_cnf
import utils.circuit_utils as circuit_utils
from utils.aiger_utils import aig_to_xdata, xdata_to_cnf

def env_map_solve(args, mapper, aig_filename, tmp_dir):
    # Map to LUTs
    ckt_name = aig_filename.split('/')[-1].split('.')[0]
    bench_filename = os.path.join(tmp_dir, '{}.bench'.format(ckt_name))
    cmd_maplut = '{} {} {}'.format(
        mapper, aig_filename, bench_filename
    )
    _, maplut_runtime = run_command(cmd_maplut, args.max_solve_time)
    if maplut_runtime < 0:
        return -1, 0, 0, 0, 0, 0
    
    # Parse LUTs
    cnf, no_var = parse_bench_cnf(bench_filename)
    # print(bench_filename)
    # exit(0)
    os.remove(bench_filename)
    
    # Solve 
    if maplut_runtime > args.max_solve_time:
        sat_res = -1 
        no_dec = 0
    else:
        sat_res, asg, solvetime, no_dec = kissat_solve_dec(args.kissat_path, cnf, no_var, tmp_dir, args='--time={:}'.format(int(args.max_solve_time - maplut_runtime)))
    
    return sat_res, maplut_runtime, solvetime, no_dec, no_var, len(cnf)

class solve_Env(gym.Env):
    def __init__(
        self,
        args, 
        config, 
        instance_list = [], 
        mode='train'
    ):
        self.instance_list = instance_list
        self.args = args
        self.config = config
        self.problem_list = []
        self.mode = mode
        self.step_ntk_filepath = self.args.step_ntk_filepath
        if len(instance_list) == 0:
            self.problem_list = os.listdir(args.Problem_AIG_Dir)
            for idx in range(len(self.problem_list)):
                self.problem_list[idx] = os.path.join(args.Problem_AIG_Dir, self.problem_list[idx])
        else:
            for aig_name in instance_list:
                aig_path = os.path.join(args.Problem_AIG_Dir, '{}.aiger'.format(aig_name))
                if os.path.exists(aig_path):
                    self.problem_list.append(aig_path)
        self.parser = dg.AigParser()
        self.action_list = [
            'rewrite', 'rewrite -l', 'rewrite -z', 
            'resub', 'resub -l', 'resub -z', 
            'refactor', 'refactor -l', 'refactor -z', 
            'balance', 'balance -l', 'balance -d', 'balance -s', 'balance -x'
            # 'renode; strash'
        ]
        self.reset_times = 0
        self.no_instance = 0 
        
        # DeepGate
        self.ckt_encoder = dg.Model(dim_hidden=args.ckt_dim)
        self.ckt_encoder.load_pretrained()
        for param in self.ckt_encoder.parameters():
            param.requires_grad = False
        
        assert len(self.action_list) == self.args.n_action - 1
        assert len(self.problem_list) > 0
        
    def parse_graph(self, g, init_g):
        state = {}
        state['emb'] = self.init_emb
        state['area'] = len(g.x) / len(init_g.x)
        level = torch.max(g.forward_level).item()
        init_level = torch.max(init_g.forward_level).item()
        state['level'] = level / init_level
        state['edge'] = len(g.edge_index[0]) / len(init_g.edge_index[0])
        
        # AND NOT Gate
        fanin_list = []
        and_cnt = 0
        not_cnt = 0
        for gate in g.gate:
            fanin_list.append([])
            if int(gate) == 1:
                and_cnt += 1
            elif int(gate) == 2:
                not_cnt += 1
        state['and'] = and_cnt / len(g.gate)
        state['not'] = not_cnt / len(g.gate)
        
        # Balance Ratio 
        for idx in range(len(g.edge_index[0])):
            src = g.edge_index[0][idx]
            dst = g.edge_index[1][idx]
            fanin_list[dst].append(src)
        tot_br = 0
        for idx in range(len(g.x)):
            if int(g.gate[idx]) == 1:
                node_a = fanin_list[idx][0]
                node_b = fanin_list[idx][1]
                tot_br += abs(g.forward_index[node_a] - g.forward_index[node_b]) / level
        state['br'] = float(tot_br / and_cnt)
        
        if self.args.large_feature:
            attr = torch.tensor([[state['area']] * 64 + [state['level']] * 64 + [state['edge']] * 64 + [state['and']] * 64 + [state['not']] * 64 + [state['br']] * 64])
        else:
            attr = torch.tensor([state['area'], state['level'], state['edge'], state['and'], state['not'], state['br']]).unsqueeze(0)
        res = torch.cat([state['emb'], attr], dim=1)
        
        return res
    
    def next_instance(self):
        if self.no_instance >= len(self.problem_list):
            self.no_instance -= len(self.problem_list)
        curr_problem = self.problem_list[self.no_instance]
        problem_name = curr_problem.split('/')[-1].split('.')[0]
        shutil.copyfile(curr_problem, self.step_ntk_filepath)
        init_cmd = 'abc -c \"read_aiger {}; rewrite -lz; balance; rewrite -lz; balance; rewrite -lz; balance; write_aiger {}; \"'.format(
            self.step_ntk_filepath, self.step_ntk_filepath
        )
        _, _ = run_command(init_cmd)
        self.origin_problem = curr_problem
        self.init_graph = self.parser.read_aiger(self.step_ntk_filepath)
        if self.args.RL_mode == 'test':
            self.sat = -1
            map_time = 0
            solve_time = 0
            no_dec = 1
            nvars = 0
            nclas = 0
        else:
            self.sat, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.args.customized_mapper, self.step_ntk_filepath, self.args.tmp_dir)
        self.bl_dec = no_dec
        self.bl_nvars = nvars
        self.bl_nclas = nclas
        self.step_cnt = 0
        self.action_time = 0
        self.problem_name = problem_name
        self.bl_st = solve_time
        self.bl_mp = map_time

        start_time = time.time()
        hs, hf = self.ckt_encoder(self.init_graph)
        self.init_emb = torch.cat([hs[self.init_graph.POs], hf[self.init_graph.POs]], dim=1)
        self.model_time = time.time() - start_time
        self.no_instance += 1
                
    def reset(self):
        self.graph = self.init_graph
        self.step_cnt = 0
        self.action_time = 0
        
        return self.parse_graph(self.graph, self.init_graph)
    
    def step(self, action):
        if action < self.args.n_action - 2 and self.step_cnt < self.args.max_step:
            action_str = self.action_list[action]
            action_cmd = 'abc -c \"read_aiger {}; {}; write_aiger {}; \"'.format(
                self.step_ntk_filepath, action_str, self.step_ntk_filepath
            )
            _, action_runtime = run_command(action_cmd)
            self.action_time += action_runtime
            self.graph = self.parser.read_aiger(self.step_ntk_filepath)
            reward = 0
            done = False
        
        else:
            self.sat_res, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.args.customized_mapper, self.step_ntk_filepath, self.args.tmp_dir)
            self.synmap_nvars = nvars
            self.synmap_nclas = nclas
            reward = (self.bl_dec - no_dec) / self.bl_dec
            done = True
            self.md_st = solve_time
            self.md_mp = self.action_time + map_time
            self.md_dec = no_dec
            self.md_nvars = nvars
            self.md_nclas = nclas
            
        self.step_cnt += 1
        info = {}
        return self.parse_graph(self.graph, self.init_graph), reward, done, info
    
    def get_solve_info(self):
        res_dict = {
            'bl_st': self.bl_st,
            'bl_mp': self.bl_mp, 
            'bl_dec': self.bl_dec,
            'bl_nvars': self.bl_nvars,
            'bl_nclas': self.bl_nclas,
            'md_st': self.md_st,
            'md_mp': self.md_mp,
            'md_dec': self.md_dec,
            'md_nvars': self.md_nvars,
            'md_nclas': self.md_nclas,
            'md_mt': self.model_time,
            'res': self.sat_res
        }
        return res_dict