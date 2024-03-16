import os
import shutil
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DeepGate.')

    # basic experiment setting
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--spc_exp_id', default='')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')
    parser.add_argument('--pretrained_path', default='../exp/pretrained/pretrain_l1/model_last.pth', type=str)
    parser.add_argument('--train_times', default=10000, type=int)
    parser.add_argument('--save_epoch', default=100)
    parser.add_argument('--load_pretrain', default='', type=str)

    # RL Q Net
    parser.add_argument('--RL_mode', default='train', choices=['test', 'train'])
    parser.add_argument('--n_action', default=15, type=int)
    parser.add_argument('--max_step', default=10, type=int)
    parser.add_argument('--ckt_dim', default=128, type=int)
    parser.add_argument('--cmd_dim', default=64, type=int)
    parser.add_argument('--mlp_dim', default=128, type=int)
    parser.add_argument('--mlp_layers', default=5, type=int)
    parser.add_argument('--large_feature', default=False, action='store_true')
    
    # Env
    parser.add_argument('--kissat_path', default='./kissat/build/kissat', type=str)
    parser.add_argument('--customized_mapper', default='./mockturtle/build/examples/my_mapper', type=str)
    parser.add_argument('--baseline_mapper', default='./mockturtle/build/examples/my_baseline', type=str)
    parser.add_argument('--min_solve_time', default=10, type=int)
    parser.add_argument('--max_solve_time', default=100, type=int)
    
    # experiment 
    parser.add_argument('--disable_encode', action='store_true', default=False)
    parser.add_argument('--disable_rl', action='store_true', default=False)

    # system
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    parser.add_argument('--random-seed', type=int, default=208, 
                             help='random seed')

    # log
    parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')

    # dataset settings
    parser.add_argument('--Problem_AIG_Dir', default='../dataset/LEC/all_case/', type=str)
    parser.add_argument('--no_rc', default=False, action='store_true')
    parser.add_argument('--data_dir', default='../data/random_circuits',
                             type=str, help='the path to the dataset')
    parser.add_argument('--enable_aig', default=True, action='store_true')      # default enable aig, no support MIG now 
    parser.add_argument('--test_data_dir', default=None,
                             type=str, help='the path to the testing dataset')
    parser.add_argument('--reload_dataset', default=False, action='store_true', help='Reload inmemory data')
    # circuit
    parser.add_argument('--gate_types', default='*', type=str,
                             metavar='LIST', help='gate types in the circuits. For aig: INPUT,AND,NOT, For Circuit-sat: INPUT,AND,OR,NOT')
    parser.add_argument('--no_node_cop', default=False, 
                             action='store_true', help='not to use the C1 values as the node features')
    parser.add_argument('--node_reconv', default=False, 
                             action='store_true', help='use the reconvergence info as the node features')
    parser.add_argument('--predict_diff', default=False, 
                             action='store_true', help='predict the difference between the simulated ground-truth probability and C1.')
    parser.add_argument('--diff_multiplier', default=10, 
                             type=int, help='the multiplier for the difference between the simulated ground-truth probability and C1.')
    parser.add_argument('--reconv_skip_connection', default=False, 
                             action='store_true', help='construct the skip connection between source ndoe and the reconvergence node.')
    parser.add_argument('--use_logic_diff', default=False, 
                             action='store_true', help='use the logic difference between the source node and the reconvergence node as the edge attributes.')
    parser.add_argument('--dim_edge_feature', default=16,
                             type=int, help='the dimension of node features')
    parser.add_argument('--logic_diff_embedding', default='positional', 
                             type=str, choices=['positional'],help='the embedding for the logic difference, only support positional embedding.')
    parser.add_argument('--logic_implication', default=False, 
                             action='store_true', help='use the logic implication/masking as an additonal node feature or not.')
    parser.add_argument('--small_train', default=False, 
                             action='store_true',help='if True, use a smaller version of train set')
    parser.add_argument('--un_directed', default=False, action='store_true', 
                             help='If true, model the circuit as the undirected graph. Default: circuit as DAG')
    # sat
    parser.add_argument('--n_pairs', default=10000, type=int, 
                             help='number of sat/unsat problems to generate')
    parser.add_argument('--min_n', type=int, default=3, 
                             help='min number of variables used for training')
    parser.add_argument('--max_n', type=int, default=10, 
                             help='max number of variables used for training')
    # neurosa
    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)
    # circuitsat
    parser.add_argument('--exp_depth', type=int, default=3)
    # deepgate-sat
    parser.add_argument('--use_aig', action='store_true', 
                             help='whether to use AIG.')
    
    # model settings
    parser.add_argument('--arch', default='mlpgnn', choices=['recgnn', 'convgnn', 'dagconvgnn', 'mlpgnn', 'mlpgnn_merge'],
                             help='model architecture. Currently support'
                                  'recgnn | convgnn ' 
                                  'recgnn will updata the embedding in T(time) dim, while convgnn will update the embedding in K(layer) dim.'
                                  'recgnn corresponds to dagnn/dvae settings, which considers DAG circuits.')
    parser.add_argument('--activation_layer', default='relu', type=str, choices=['relu', 'relu6', 'sigmoid'],
                             help='The activation function to use in the FC layers.')  
    parser.add_argument('--norm_layer', default='batchnorm', type=str,
                             help='The normalization function to use in the FC layers.')
    parser.add_argument('--num_fc', default=3, type=int,
                             help='The number of FC layers')                          
    # recgnn 
    parser.add_argument('--num_aggr', default=3, type=int,
                             help='the number of aggregation layers.')
    parser.add_argument('--aggr_function', default='tfmlp', type=str, choices=['deepset', 'aggnconv', 'gated_sum', 'conv_sum', 'mlp', 'attnmlp', 'tfmlp'],
                             help='the aggregation function to use.')
    parser.add_argument('--update_function', default='gru', type=str, choices=['gru', 'lstm'],
                             help='the update function to use.')
    parser.add_argument('--wx_update', action='store_true', default=False,
                            help='The inputs for the update function considers the node feature of mlp.')
    parser.add_argument('--no_keep_input', action='store_true', default=False,
                             help='no to use the input feature as the input to recurrent function.')
    parser.add_argument('--aggr_state', action='store_true', default=False,
                             help='use the aggregated message as the previous state of recurrent function.')
    parser.add_argument('--init_hidden', action='store_true', 
                             default=False, help='whether to init the hidden state of node embeddings')
    parser.add_argument('--num_rounds', type=int, default=1, metavar='N',
                             help='The number of rounds for grn propagation.'
                             '1 - the setting used in DAGNN/D-VAE')
    parser.add_argument('--intermediate_supervision', action='store_true', default=False,
                             help='Calculate the losses for every round.')
    parser.add_argument('--mask', action='store_true', default=False,
                             help='Use the mask for the node embedding or not')
    parser.add_argument('--no_reverse', action='store_true', default=False,
                             help='Not to use the reverse layer to propagate the message.')
    parser.add_argument('--custom_backward', action='store_true', default=False,
                             help='Whether to use the custom backward or not.')
    parser.add_argument('--seperate_hidden', action='store_true', default=False,
                             help='seperate node hidden states for forward layer and backward layer.')
    parser.add_argument('--dim_hidden', type=int, default=64, metavar='N',
                             help='hidden size of recurrent unit.')
    parser.add_argument('--dim_mlp', type=int, default=32, metavar='N',
                             help='hidden size of readout layers') 
    parser.add_argument('--dim_pred', type=int, default=1, metavar='N',
                             help='hidden size of readout layers')
    parser.add_argument('--mul_mlp', action='store_true', default=False,
                             help='To use seperate MLP for different gate types.') 
    parser.add_argument('--wx_mlp', action='store_true', default=False,
                             help='The inputs for the mlp considers the node feature of mlp.')        
    # convgnn

    # circuitsat/deepsat
    parser.add_argument('--temperature', type=float, default=0.01,
                             help='initial value for temperature')
    parser.add_argument('--eplison', type=float, default=0.4,
                             help='the anneling factore of temperature.')
    parser.add_argument('--k_step', type=float, default=10.0,
                             help='the value for step funtion parameter k.')
    parser.add_argument('--prob_loss', action='store_true', default=False,
                             help='To use the simulated probabilities as complementary supervision.')
    parser.add_argument('--prob_weight', type=float, default=0.1,
                             help='the weight for simulated probability loss.')                       
             
                     
    # loss
    parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2 | focalloss')
    parser.add_argument('--cls_loss', default='bce',
                             help='classification loss: bce - BCELoss | bce_logit - BCELossWithLogit | cross - CrossEntropyLoss')
    parser.add_argument('--sat_loss', default='smoothstep', choices=['smoothstep'],
                             help='the loss for circuitsat: smoothstep')
    parser.add_argument('--Prob_weight', type=float, default=5)
    parser.add_argument('--RC_weight', type=float, default=3)
    parser.add_argument('--Func_weight', type=float, default=1)


    # train and val
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-10, 
                             help='weight decay (default: 1e-10)')
    parser.add_argument('--lr_step', type=str, default='30,45',
                             help='drop learning rate by 10.')
    parser.add_argument('--grad_clip', type=float, default=0.,
                        help='gradiant clipping')
    parser.add_argument('--num_epochs', type=int, default=40,
                             help='total training epochs.')
    parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    parser.add_argument('--trainval_split', default=0.9, type=float,
                             help='the splitting setting for training dataset and validation dataset.')
    parser.add_argument('--val_only', action='store_true', 
                             help='Do the validation evaluation only.')
    
    # test
    parser.add_argument('--test_split', default='test', choices=['test', 'train', 'all'],
                             help='the split to use for testing.')
    parser.add_argument('--cop_only', action='store_true',
                             help='only show the comparision between C1 and simluated probability.')
    # parser.add_argument('--test_num_rounds', default=10, type=int,
    #                          help='The number of rounds to be run during testing.')

    

    args = parser.parse_args()

    args.lr_step = [int(i) for i in args.lr_step.split(',')]


    # update data settings
    if args.enable_aig:
        args.gate_to_index = {'PI': 0, 'AND': 1, 'NOT': 2}
    else:
        args.gate_to_index = {'PI': 0, 'GND': 1, 'VDD': 2, 'MAJ': 3, 'NOT': 4, 'BUF': 5}
    args.num_gate_types = len(args.gate_to_index)
    args.dim_node_feature = len(args.gate_to_index)

    # check the relationship of `task`, `dataset` and `arch` comply with each other. TODO: optimize this part
    args.circuit_file = "graphs.npz"
    args.label_file = "labels.npz"

    if args.use_logic_diff:
        assert args.logic_diff_embedding == "positional", "Only support positional embedding for the logic difference." 
        assert args.reconv_skip_connection, "Using logic differce as the edge attributes is activated when we build the skip connection between source node and reconvegence node."
    args.use_edge_attr = args.reconv_skip_connection and args.use_logic_diff

    args.reverse = not args.no_reverse


    # assert args.dim_node_feature == (len(args.gate_to_index)) + int(not args.no_node_cop) + int(args.node_reconv) + int(args.logic_implication), "The dimension of node feature is not consistent with the specification, please check it again." 
    # assert args.dim_node_feature == (len(args.gate_to_index)) + int(not args.no_node_cop) + int(args.node_reconv), "The dimension of node feature is not consistent with the specification, please check it again." 
    
    if args.predict_diff:
        assert args.no_node_cop, "Predicting the different of C1 and gt, and including COP into node features cannot be combined together"


    if args.debug > 0:
        args.num_workers = 0

    if args.spc_exp_id != '':
        args.exp_id = args.spc_exp_id

    # dir
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp')
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    args.tmp_dir = os.path.join(args.save_dir, 'tmp')
    args.step_ntk_filepath = os.path.join(args.tmp_dir, 'step_ntk.aig')
    update_dir(args)
    
    if args.load_pretrain != '':
        args.resume = True
        model_path = os.path.join(args.save_dir, 'qnet_last.pth')
        if os.path.exists(model_path):
            os.remove(model_path)
        shutil.copy(args.load_pretrain, model_path)

    args.local_rank = 0

    return args

def update_dir(args):
    # dir
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp')
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    args.tmp_dir = os.path.join(args.save_dir, 'tmp')
    print('The output will be saved to ', args.save_dir)
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.debug_dir):
        os.mkdir(args.debug_dir)
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)

    if args.resume and args.load_model == '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, 'model_last.pth')
    elif args.load_model != '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, args.load_model)
    
    return args
