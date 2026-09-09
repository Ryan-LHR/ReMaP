# -*-coding:utf-8-*-
import os
import sys
import ast
import argparse
import time

from termcolor import colored

from utils import get_datetime
from utils.models import *

def parser_add_args(parser):
    """Define parser"""
    # General
    parser.add_argument("-seed", action="store", default=0, type=int)  # random seed
    parser.add_argument("-seed_list", nargs="+", type=int)  # random seeds
    parser.add_argument("-dataset", action="store", default='fashion_mnist',
                        type=str, help='fashion_mnist, cifar_10')  # dataset
    parser.add_argument("-model", action="store", default=None, type=str)  # model
    parser.add_argument("-model_class", type=bool, default=None)  # define class of model
    parser.add_argument("-device", action="store", default='cpu', type=str)  # device
    parser.add_argument("-rq", action="store", default='rq1', type=str)  # which experiment
    parser.add_argument("-n_workers", type=int, default=1)

    # Path
    parser.add_argument("-data_dir", action="store", type=str)  # directory of dataset, model
    parser.add_argument("-save_dir", action="store", type=str)  # directory of results
    parser.add_argument("-save_result", action="store", default='False')
    parser.add_argument("-load_prediction", action="store", default='False')  # load saved prediction results
    parser.add_argument("-load_feature", action="store", default='False')  # load saved features
    parser.add_argument("-load_hparam", action="store", default='True')  # load auto-selected beta/omega
    parser.add_argument("-load_reliability", action="store", default='True')  # set False when evaluating efficiency

    # Selection
    parser.add_argument("-method", action="store", type=str)  # Selection method
    parser.add_argument("-strategy", action="store", type=str)  # strategy of coverage based method

    parser.add_argument("-budget_list", action="store", type=str)  # budgets for selection method
    parser.add_argument("-budget", action="store", type=float)  # budgets for single retrain
    parser.add_argument("-cand_type", action="store", type=str)  # type for candidate set
    parser.add_argument("-cand_size", action="store")  # size for candidate set

    # RQ-Retrain specific args
    parser.add_argument("-load_selection_result", action="store", default='False')  # load selection result for retraining
    parser.add_argument("-split_seed_list", action="store", default='[0]', type=str)  # random seeds for dataset split

    # Training
    parser.add_argument("-shuffle", type=bool, default=True)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-lr", type=float, default=None)
    parser.add_argument("-scheduler", type=int, default=None)
    parser.add_argument("-criterion", type=str, default=None)

    # Config
    parser.add_argument("-machine", type=str, default='local')
    parser.add_argument("-specific_seeds", action="store", default=None, type=str)  # for missing seed
    parser.add_argument("-debug_mode", action="store", default='False')  # debug
    parser.add_argument("-downsample_ratio", type=float, default=0.1)

    args = parser.parse_args()
    print("Parsed arguments:", vars(args))  # print all arguments
    return args

def modify_args(args):
    """Get detailed args"""
    # Get machine info
    args.machine_id = get_machine_id()
    args.machine_name = get_machine_name(args.machine_id)
    args.machine = get_machine(args.machine_name)

    if args.machine == 'local':
        args.data_dir = "../data"
        model_dir = '../data/models'
    elif args.machine == 'autodl':
        args.data_dir = '/root/autodl-tmp/Projects/dimp/data'
        model_dir = '/root/autodl-tmp/Projects/dimp/data/models'
    else:
        raise ValueError("Machine Not Found.")

    # Get model file path and model class
    if args.model == 'FM-ResNet20':
        args.model_file = 'FM-ResNet20/FM-ResNet20.pth'
        args.model_class = FM_ResNet20
    elif args.model == 'MNIST-LeNet5':
        args.model_file = 'MNIST-LeNet5/MNIST-LeNet5.pth'
        args.model_class = MNIST_LeNet5
    elif args.model == 'C10-ResNet20':
        args.model_file = 'C10-ResNet20/C10-ResNet.pth'
        args.model_class = C10_ResNet20
    elif args.model == 'SVHN-VGG16':
        args.model_file = 'SVHN-VGG16/SVHN-VGG16.pth'
        args.model_class = VGG16
    elif args.model == "IM100Test-deit_base_patch16_224":
        args.model_file = 'IM100Test/deit-base-patch16-224'
    elif args.model == "ModelNet40-DGCNN":
        args.model_file = 'ModelNet40-DGCNN/ModelNet40-DGCNN.pth'
        args.model_class = ModelNet40_DGCNN
    elif args.model == "ESC50-AST":
        # public HF repo id (loaded via hub);
        # AST fine-tuned on ESC-50, fold 1 held out (test), fold 2 val, folds 3-5 train, 93.5% test acc
        args.model_file = 'Adam-ousse/ast-esc50-finetuned-fold1'
        args.model_class = None
    else:
        raise ValueError("Model Not Found.")
    if args.model != "ESC50-AST":
        # hub-loaded audio models keep their HF repo id (not a local model_dir path)
        args.model_file = os.path.join(model_dir, args.model_file)

    # Get dataset name
    if args.model == 'FM-ResNet20':
        args.dataset = 'fashion_mnist'
    elif args.model == 'MNIST-LeNet5':
        args.dataset = 'mnist'
    elif args.model == 'C10-ResNet20':
        args.dataset = 'cifar_10'
    elif args.model == 'SVHN-VGG16':
        args.dataset = 'svhn'
    elif args.model == "IM100Test-deit_base_patch16_224":
        args.dataset = 'imagenet_100'
        args.batch_size = 128
    elif args.model == "ModelNet40-DGCNN":
        args.dataset = 'modelnet40'
        args.batch_size = 32  # DGCNN edge-conv [B,N,k,2C] hits ~5 GB at B=256; 32 keeps single-alloc <1 GB
    elif args.model == "ESC50-AST":
        args.dataset = 'esc50'
        args.batch_size = 32
    else:
        raise ValueError("Dataset Not Found")

    # set random seeds
    # RANDOM_METHODS = ["random",
    #                   "nac-cam", "nbc-cam", "snac-cam", "kmnc-cam", "tknc-cam",
    #                   "pc-mlsa", "pc-dsa", "pc-mmdsa",
    #                   "rts", "certpri"]
    RANDOM_METHODS = [None]
    if args.rq == "rq1-priorization":
        # if args.method in RANDOM_METHODS:
        #     args.seed_list = [0, 1, 2, 3, 4]
        #     # args.seed_list = [0, 1]
        #     # args.seed_list = range(5)
        # else:
        #     args.seed_list = [0]
        #     # args.seed_list = [2]
        args.seed_list = [0, 1, 2, 3, 4]  # all repeat 5 times

        if args.specific_seeds != "None":
            args.seed_list = ast.literal_eval(args.specific_seeds)
            print(colored(f'\nTemporary seed_list is {args.specific_seeds}\n',
                          'red', attrs=['bold', 'underline']),)

    args.split_seed_list = ast.literal_eval(args.split_seed_list)

    args.dataset_dir = os.path.join(args.data_dir, 'datasets')
    args.model_dir = os.path.join(args.data_dir, 'models')
    args.budget_list = ast.literal_eval(args.budget_list)

    args.save_dir_result = os.path.join('../results', "{}/{}/results".format(args.rq, args.model))
    os.makedirs(args.save_dir_result, exist_ok=True)  # e.g.'../results/rq1-priorization/MNIST-LeNet5/results'

    args.ex_time = get_datetime()  # get time of experiment start

    args.save_result = args.save_result.lower() == "true"
    args.load_prediction = args.load_prediction.lower() == "true"
    args.load_feature = args.load_feature.lower() == "true"
    args.load_hparam = args.load_hparam.lower() == "true"
    args.load_reliability = args.load_reliability.lower() == "true"
    args.load_selection_result = args.load_selection_result.lower() == "true"
    args.debug_mode = args.debug_mode.lower() == "true"

    return args

def get_settings(args):
    """Return a string describe the experiment settings"""
    settings = f'RQ: {args.rq} \n' \
               f'Model: {args.model} \n' \
               f'Dataset: {args.dataset} \n' \
               f'Method: {args.method} \n' \
               f'Candidate Type: {args.cand_type} \n' \
               f'Seed: {args.seed}'

    return settings

def get_retrain_settings(args):
    """Return a string describe the retraining settings"""
    settings = f'Split Seed: {args.split_seed} \n' \
               f'Budget: {args.budget} \n' \
               f'Shuffle: {args.shuffle} \n' \
               f'Epochs: {args.epochs} \n' \
               f'Batch Size: {args.batch_size} \n' \
               f'LR: {args.lr} ' \
        # f'Optimizer: {args.optimizer} \n' \
               # f'Scheduler: {args.scheduler}'

    return settings

def get_machine_id():
    """Return the unique identification """
    import uuid, socket
    machine_id = f"{socket.gethostname()}-{uuid.getnode()}"
    # hash_id = hashlib.sha256(machine_id.encode()).hexdigest()

    print(f"Current Machine ID is: {machine_id}")
    return machine_id

def get_machine_name(machine_id):
    """Return the unique machine name """

    if "YOUR_MACHINE_ID" in machine_id:
        return "AutoDL-ReMaP"
    else:
        return "local"

def get_machine(machine_name):
    """Return"""
    if "Auto" in machine_name:
        return "autodl"
    else:
        return "local"
