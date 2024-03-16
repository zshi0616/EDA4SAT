import enum
import os
import utils.utils as utils
import random
import numpy as np
from datetime import datetime
import copy

def get_hash(lst):
    hash_value = hash(tuple(lst)) % (1e9 + 7)
    return int(hash_value)

def parse_solution(solve_info, no_vars):
    asg = [0] * no_vars
    for idx, line in enumerate(solve_info):
        if 'Learnt' in line:
            continue
        if line[0] == 's':
            continue
        if 'v ' not in line:
            continue
        arr = line.replace('v ', '').replace('\n', '').split(' ')
        for ele in arr:
            num = int(ele)
            if num == 0:
                break
            elif num > 0:
                asg[num - 1] = 1
            else:
                asg[abs(num) - 1] = 0
    return asg

def save_cnf(iclauses, n_vars, filename): 
    n_clauses = len(iclauses)
    f = open(filename, 'w')
    # head
    f.write('p cnf {:} {:}\n'.format(n_vars, n_clauses))

    # CNF
    for clause in iclauses:
        new_line = ''
        for ele in clause:
            new_line += str(ele) + ' '
        new_line += '0\n'
        f.write(new_line)
    
    f.close()
    
def save_bench(iclauses, n_vars, filename):
    f = open(filename, 'w')
    for pi_idx in range(1, n_vars + 1):
        f.write('INPUT({:})\n'.format(pi_idx))
    f.write('OUTPUT(PO)\n')
    f.write('\n')
    for pi_idx in range(1, n_vars + 1):
        f.write('{:}_INV = NOT({:})\n'.format(pi_idx, pi_idx))
    for clause_idx, clause in enumerate(iclauses):
        newline = 'CLAUSE_{:} = OR('.format(clause_idx)
        for var_idx, var in enumerate(clause):
            if var > 0:
                newline += '{:}'.format(var)
            else:
                newline += '{:}_INV'.format(abs(var))
            if var_idx == len(clause) - 1:
                newline += ')\n'
            else:
                newline += ', '
        f.write(newline)
    newline = 'PO = AND('
    for clause_idx in range(len(iclauses)):
        if clause_idx == len(iclauses) - 1:
            newline += 'CLAUSE_{:})\n'.format(clause_idx)
        else:
            newline += 'CLAUSE_{:}, '.format(clause_idx)
    f.write(newline)
    f.close()

def kissat_solve(iclauses, no_vars, tmp_filename=None, args=None):
    if tmp_filename == None:
        tmp_filename = './tmp/tmp_solve_{:}_{:}_{:}_{:}_{:}.cnf'.format(
            datetime.now().hour, datetime.now().minute, datetime.now().second, len(iclauses), random.randint(0, 10000)
        )
    
    save_cnf(iclauses, no_vars, tmp_filename)
    if args != None:
        cmd_solve = '{} {} -q {}'.format('kissat', args, tmp_filename)
    else:
        cmd_solve = '{} -q {}'.format('kissat', tmp_filename)
        
    solve_info, solvetime = utils.run_command(cmd_solve)
    # Check satisfibility
    is_sat = True
    is_unknown = True
    for line in solve_info:
        if 'UNSATISFIABLE' in line:
            is_unknown = False
            is_sat = False
            break
        if 'SATISFIABLE' in line:
            is_unknown = False
    os.remove(tmp_filename)

    if is_sat and not is_unknown:
        asg = parse_solution(solve_info, no_vars)
        # asg = ['SAT']
    else:
        asg = []
    # Sat Status: SAT=1, UNSAT=0, UNKNOWN=-1
    if is_unknown:
        sat_status = -1
    elif is_sat:
        sat_status = 1
    else:
        sat_status = 0
        
    return sat_status, asg, solvetime

def kissat_solve_dec(kissat_path, iclauses, no_vars, tmp_dir, args=None):
    tmp_filename = os.path.join(tmp_dir, 'tmp_solve_{:}_{:}_{:}_{:}_{:}.cnf'.format(
        datetime.now().hour, datetime.now().minute, datetime.now().second, len(iclauses), random.randint(0, 10000)
    ))
    
    save_cnf(iclauses, no_vars, tmp_filename)
    if args != None:
        cmd_solve = '{} {} -q {}'.format(kissat_path, args, tmp_filename)
    else:
        cmd_solve = '{} -q {}'.format(kissat_path, tmp_filename)
        
    solve_info, solvetime = utils.run_command(cmd_solve)
    # Check satisfibility
    is_sat = True
    is_unknown = True
    for line in solve_info:
        if 'UNSATISFIABLE' in line:
            is_unknown = False
            is_sat = False
            break
        if 'SATISFIABLE' in line:
            is_unknown = False
    os.remove(tmp_filename)
    
    no_dec = -1
    if not is_unknown:
        for line in solve_info:
            if 'DEC' in line:
                no_dec = int(line.replace('\n', '').replace(' ', '').split(':')[-1])
                break
            
    if is_sat and not is_unknown:
        asg = parse_solution(solve_info, no_vars)
        # asg = ['SAT']
    else:
        asg = []
    # Sat Status: SAT=1, UNSAT=0, UNKNOWN=-1
    if is_unknown:
        sat_status = -1
    elif is_sat:
        sat_status = 1
    else:
        sat_status = 0
        
    return sat_status, asg, solvetime, no_dec

def read_cnf(cnf_path):
    f = open(cnf_path, 'r')
    lines = f.readlines()
    f.close()

    n_vars = -1
    n_clauses = -1
    begin_parse_cnf = False
    iclauses = []
    for line in lines:
        if begin_parse_cnf:
            arr = line.replace('\n', '').split(' ')
            clause = []
            for ele in arr:
                if ele.replace('-', '').isdigit() and ele != '0':
                    clause.append(int(ele))
            if len(clause) > 0:
                iclauses.append(clause)
                
        elif line.replace(' ', '')[0] == 'c':
            continue
        elif line.replace(' ', '')[0] == 'p': 
            arr = line.replace('\n', '').split(' ')
            get_cnt = 0
            for ele in arr:
                if ele == 'p':
                    get_cnt += 1
                elif ele == 'cnf':
                    get_cnt += 1
                elif ele != '':
                    if get_cnt == 2:
                        n_vars = int(ele)
                        get_cnt += 1
                    else: 
                        n_clauses = int(ele)
                        break
            assert n_vars != -1
            assert n_clauses != -1
            begin_parse_cnf = True
        
    
    return iclauses, n_vars

def divide_cnf(cnf, no_vars, no_sub_cnfs):
    mark_list = [0] * len(cnf)
    for k in range(no_sub_cnfs - 1):
        mark_list[k] = 1
    random.shuffle(mark_list)
    
    sub_cnf_list = []
    sub_cnf = []
    for clause_idx in range(len(cnf)):
        if mark_list[clause_idx] == 1:
            sub_cnf_list.append(sub_cnf)
            sub_cnf = []
        sub_cnf.append(cnf[clause_idx])
    
    sub_cnf_list.append(sub_cnf)
    return sub_cnf_list

def get_sub_cnf(cnf, var, is_inv):
    res_cnf = []
    if not is_inv:
        for clause in cnf:
            if not var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == -var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    else:
        for clause in cnf:
            if not -var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    return res_cnf
        
def unit_prop(cnf, assign):
    new_cnf = []
    for clause in cnf:
        new_clause = []
        satisfied = False
        for var in clause:
            if assign == var:
                satisfied = True
                break
            elif assign == -1 * var:
                continue
            else:
                new_clause.append(var)
        if not satisfied:
            new_cnf.append(new_clause)
    return new_cnf

def simulation(cnf, no_vars):
    assignment = []
    ass_hash_list = []
    for pattern_idx in range(0, int(pow(2, no_vars))):
        res_cnf = copy.deepcopy(cnf)
        pattern = bin(pattern_idx)[2:].zfill(no_vars)
        pattern = [int(ele) for ele in pattern]
        for idx, ele in enumerate(pattern):
            var = idx + 1
            var_assign = var if ele == 1 else -var
            res_cnf = unit_prop(res_cnf, var_assign)
        if len(res_cnf) == 0:
            ass_hash = get_hash(pattern)
            if ass_hash not in ass_hash_list:
                assignment.append(pattern)
                ass_hash_list.append(ass_hash)
    return assignment

def resolve(cnf):
    new_cnf = copy.deepcopy(cnf)
    cover_list = []
    origin_size = len(new_cnf)
    clause_hash_dict = {}
    for idx, clause in enumerate(new_cnf):
        hash_value = get_hash(clause)
        if hash_value not in clause_hash_dict.keys():
            clause_hash_dict[hash_value] = 1
        cover_list.append([idx])
        
    while True:
        resolved = False
        i = 0
        while i < len(new_cnf):
            cnf_i = new_cnf[i]
            i += 1
            j = i
            while j < len(new_cnf):
                cnf_j = new_cnf[j]
                j += 1
                # Find resolvent
                reverse_lit = 0
                for l in cnf_i:
                    if -l in cnf_j: 
                        reverse_lit = l
                        break
                if reverse_lit != 0:
                    resolvent = [l for l in cnf_i if l != reverse_lit] + [l for l in cnf_j if l != -reverse_lit]
                    var_dict = {}
                    for lit in resolvent:
                        if lit in var_dict:
                            var_dict[lit] += 1
                        else:
                            var_dict[lit] = 1
                    is_success_resolvent = True
                    for lit in list(var_dict.keys()):
                        if -lit in var_dict:
                            is_success_resolvent = False
                            break
                    resolvent = list(var_dict.keys())
                    if len(resolvent) == 0:
                        is_success_resolvent = False
                        
                    hash_value = get_hash(resolvent)
                    if is_success_resolvent and hash_value not in clause_hash_dict.keys():
                        resolved = True
                        new_cnf.append(resolvent)
                        cover_list.append(list(set(cover_list[i-1] + cover_list[j-1])))
                        # cover_list.append([i-1, j-1])
                        clause_hash_dict[hash_value] = 1
                        
        if resolved == False:
            break
    return new_cnf
