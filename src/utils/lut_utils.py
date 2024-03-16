'''
Utility functions for Look-up-table
'''
import time
import copy
import numpy as np 
import os 
from numpy.random import randint
from collections import Counter

from utils.utils import run_command
from utils.cnf_utils import kissat_solve

def read_file(file_name):
    f = open(file_name, "r")
    data = f.readlines()
    return data

def feature_gen_init(lines): 
    node2idx = {}
    x_data = []
    fanin_list = []
    fanout_list = []
    
    # Find node names 
    for line in lines: 
        if 'INPUT' in line:
            node_name = line.split("(")[1].split(")")[0]
            node2idx[node_name] = len(x_data)
            x_data.append([node_name, ''])
        elif 'LUT' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            func = tmp_line.split('LUT')[-1].split('(')[0]
            node2idx[node_name] = len(x_data)
            x_data.append([node_name, func])
        elif 'gnd' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            func = 'gnd'
            node2idx[node_name] = len(x_data)
            x_data.append([node_name, func])
        elif 'vdd' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            func = 'vdd'
            node2idx[node_name] = len(x_data)
            x_data.append([node_name, func])
            
    no_nodes = len(x_data)
    for idx in range(no_nodes):
        fanin_list.append([])
        fanout_list.append([])
            
    for line in lines:
        if 'LUT' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            dst_idx = node2idx[node_name]
            fanin_line = tmp_line.split('(')[-1].split(')')[0]
            fanin_name_list = fanin_line.split(',')
            for fanin_name in fanin_name_list:
                fanin_idx = node2idx[fanin_name]
                fanin_list[dst_idx].append(fanin_idx)
                fanout_list[fanin_idx].append(dst_idx)
        elif 'gnd' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            dst_idx = node2idx[node_name]
            fanin_list[dst_idx] = [-1]
        elif 'vdd' in line:
            tmp_line = line.replace(' ', '')
            node_name = tmp_line.split('=')[0]
            dst_idx = node2idx[node_name]
            fanin_list[dst_idx] = [-1]
                
    return x_data, fanin_list, fanout_list

def convert_cnf(data, fanin_list, po_idx=-1):
    cnf = []
    for idx, x_data_info in enumerate(data): 
        if x_data_info[1] == '':
            continue
        if x_data_info[1] == 'gnd':
            cnf.append([-1 * (idx + 1)])
            continue
        if x_data_info[1] == 'vdd':
            cnf.append([1 * (idx + 1)])
            continue
        tt_len = int(pow(2, len(fanin_list[idx])))
        func = bin(int(x_data_info[1], 16))[2:].zfill(tt_len)
        # print(x_data_info[1], tt_len)
        
        for func_idx, y_str in enumerate(func):
            y = 1 if int(y_str) == 1 else -1
            clause = [y * (idx+1)]
            
            mask_val = tt_len - func_idx - 1
            mask_list = bin(mask_val)[2:].zfill(len(fanin_list[idx]))
            for k, ele in enumerate(mask_list):
                var = fanin_list[idx][-1 * (k+1)] + 1
                var = var if int(ele) == 0 else (-1 * var)
                clause.append(var)
            cnf.append(clause)
    if po_idx != -1:
        cnf.append([po_idx + 1])
            
    return cnf                       

def get_pi_po(fanin_list, fanout_list): 
    pi_list = []
    po_list = []
    for idx in range(len(fanin_list)):
        if len(fanin_list[idx]) == 0 and len(fanout_list[idx]) > 0:
            pi_list.append(idx)
    for idx in range(len(fanout_list)):
        if len(fanout_list[idx]) == 0 and len(fanin_list[idx]) > 0:
            po_list.append(idx)
    return pi_list, po_list

def get_level(x_data, fanin_list, fanout_list):
    bfs_q = []
    x_data_level = [-1] * len(x_data)
    max_level = 0
    for idx, x_data_info in enumerate(x_data):
        if len(fanin_list[idx]) == 0:
            bfs_q.append(idx)
            x_data_level[idx] = 0
    while len(bfs_q) > 0:
        idx = bfs_q[-1]
        bfs_q.pop()
        tmp_level = x_data_level[idx] + 1
        for next_node in fanout_list[idx]:
            if x_data_level[next_node] < tmp_level:
                x_data_level[next_node] = tmp_level
                bfs_q.insert(0, next_node)
                if x_data_level[next_node] > max_level:
                    max_level = x_data_level[next_node]
    level_list = []
    for level in range(max_level+1):
        level_list.append([])
        
    if -1 in x_data_level:
        print('Wrong')
        raise
    else:
        if max_level == 0:
            level_list = [[]]
        else:
            for idx in range(len(x_data)):
                level_list[x_data_level[idx]].append(idx)
    return level_list

def parse_bench(bench_file):
    data = read_file(bench_file)
    data, fanin_list, fanout_list = feature_gen_init(data)
    return data, fanin_list, fanout_list
    
def parse_bench_cnf(bench_file):
    data = read_file(bench_file)
    data, fanin_list, fanout_list = feature_gen_init(data)
    pi_list, po_list = get_pi_po(fanin_list, fanout_list)
    no_var = len(data)
    
    # Check PO
    po_candidates = []
    for idx in po_list:
        if data[idx][1] != 'gnd' and data[idx][1] != 'vdd':
            po_candidates.append(idx)
    if len(po_candidates) != 1:
        return [], 0
    
    cnf = convert_cnf(data, fanin_list, po_candidates[0])
    return cnf, no_var

def partition(data, fanin_list, fanout_list, level_list, partition_level): 
    low_data = []
    org2low = {}
    low_fanin_list = []
    low_fanout_list = []
    high_data = []
    org2high = {}
    high_fanin_list = []
    high_fanout_list = []
    
    for level in range(len(level_list)):
        if level <= partition_level: 
            for org_idx in level_list[level]:
                low_idx = len(low_data)
                org2low[org_idx] = low_idx
                low_data.append(low_idx)
        if level >= partition_level:
            for org_idx in level_list[level]:
                high_idx = len(high_data)
                org2low[org_idx] = high_idx
                high_data.append(high_idx)

def parse_config_formula(config, input_name_list = ['A', 'B', 'C', 'D']):
    no_input = len(input_name_list)
    tt_len = int(pow(2, no_input))
    func = bin(int(config, 16))[2:].zfill(tt_len)
    res = 'Y='
    
    for func_idx, y_str in enumerate(func):
        y = 1 if int(y_str) == 1 else -1
        mask_val = tt_len - func_idx - 1
        mask_list = bin(mask_val)[2:].zfill(no_input)
        res += '('
        for k, ele in enumerate(mask_list):
            var = input_name_list[-1*(k+1)]
            var = '!'+var if int(ele) == 0 else (var)
            res += var
            if k != len(mask_list)-1:
                res += '*'
        res += ')'
        if func_idx != len(func)-1:
            res += '+'
    res += ';'
    return res    
        
def lutmap_solve(mapper_filepath, aig_filename, tmp_dir, maxtime=-1, args=None):
    # Map to LUTs
    ckt_name = aig_filename.split('/')[-1].split('.')[0]
    bench_filename = os.path.join(tmp_dir, '{}.bench'.format(ckt_name))
    cmd_maplut = '{} {} {}'.format(
        mapper_filepath, aig_filename, bench_filename
    )
    if maxtime == -1:
        _, maplut_runtime = run_command(cmd_maplut)
    else:
        _, maplut_runtime = run_command(cmd_maplut, timeout=maxtime)
    if maplut_runtime < 0:
        return -1, 0, 0, 0, 0
    
    # Parse LUTs
    cnf, no_var = parse_bench_cnf(bench_filename)
    os.remove(bench_filename)
    
    # Solve 
    sat_res, asg, solvetime = kissat_solve(cnf, no_var, args=args)
    
    return sat_res, maplut_runtime, solvetime, no_var, len(cnf)
    