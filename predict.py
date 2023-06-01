''' Libraries '''
import argparse
import os
import torch

from exp.exp_informer import Exp_Informer


''' Parameters '''
#MODEL_DIRECTORY = "results/2021-03-28_23.20.20_bs5_sl60_ll40_pl5_fncombined_addinfo_rm_front_2.csv_lr0.05_lradj0.95_1"


''' Functions '''
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--model_path', type=str, default='results/', help='root path of the model result')
    parser.add_argument('--path', type=str, default='', help='path of the model path')
    parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
    parser.add_argument('--distil', action='store_true', help='whether to use distilling in encoder', default=False)
    parser.add_argument('--passthrough', action='store_true', help='whether to use passthrough mechanism in encoder, default=False', default=False)
    args = parser.parse_args()

    args.seq_len       = 60
    args.label_len     = 40
    args.pred_len      = 5
    args.dropout       = 0.05
    args.batch_size    = 5
    args.learning_rate = 1e-4
    args.lradj         = '0.7'
    args.data_path     = 'AIRT.csv'
    args.e_layers      = 5 # 12
    args.d_layers      = 5 # 12
    args.attn          = 'prob'
    args.factor        = 5

    args.enc_in  = 6
    args.dec_in  = 6
    args.d_model = 512
    args.d_ff    = 2048
    args.scale   = True  # Add by Stock
    # args.distil  = False
    # args.passthrough = False

    # args.model = 'informer'
    args.data = 'custom'
    args.root_path = './data/stocks_10/'
    args.features = 'MS'
    args.target = 'Close'
    args.freq = 'd'
    args.c_out = 1
    args.n_heads = 8
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 1000
    args.patience = 5
    args.loss = 'mse'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    return args


''' Execution '''
if __name__ == '__main__':

    torch.cuda.empty_cache()

    args = get_args()
    print('Args in experiment:')
    print(args)

    exp = Exp_Informer(args)
    exp.model.load_state_dict(torch.load(f"{args.model_path + args.path}/checkpoint.pth"))
    exp.model.eval()
    
    exp.predict(args.model_path + args.path)