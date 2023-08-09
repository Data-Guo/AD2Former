import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from model.cross_u_trans_ori import cross_u_trans
# from model.DeepTR import DeepTR
from trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--config_file', type=str,
                    default='checkpoint_cross_u_trans_origin_nose', help='config file name w/o suffix')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--input_channels', type=int, default=3, help='input channels')
parser.add_argument('--ce', type=float, default=0.5)
parser.add_argument('--dice', type=float, default=0.5)
# parser.add_argument('--att_block', type=str, default='ca_block')
# parser.add_argument('--view change reshape', type=str, default='view change reshape')
# parser.add_argument('--is_coutiguous', type=str, default='no_cou')
# parser.add_argument('--res', type=str, default='yesres')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.exp = 'cross_u_trans' + dataset_name + str(args.img_size)+ '_' + str(args.ce)+'ce'+'_'+str(args.dice)+'dice'+'_'+str(args.base_lr)+'base_lr'+'_'+str(args.max_epochs)
    snapshot_path = "./train_ckpt/{}/{}".format(args.exp, 'cross_u_trans_origin_nose')
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    net = cross_u_trans(args.input_channels,args.num_classes).cuda()
    # net = DeepTR().cuda()

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)