''' Libraries '''
import argparse
import os
import torch

from exp.exp_informer import Exp_Informer


''' Functions '''
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='results/', help='root path of the model result')
    parser.add_argument('--path', type=str, default='', help='path of the model path')
    
    parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--data', type=str, default='custom', help='data')
    parser.add_argument('--root_path', type=str, default='./data/stocks_10/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='AIRT.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Close', help='target feature in S or MS task')
    parser.add_argument('--scale', type=bool, default=True, help='scale the dataset (Add by Stock)')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')

    # the seq_len need to the multilply of the 2^(e_layers)
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length of Informer encoder') # 120
    parser.add_argument('--label_len', type=int, default=40, help='start token length of Informer decoder') # 60 
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length') # 15
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--e_layers', type=int, default=5, help='num of encoder layers') ## use 12
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') ## use 12

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--distil', action='store_true', help='whether to use distilling in encoder', default=False)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, log, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='0.7', help='adjust learning rate')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # TCCT
    #parser.add_argument('--CSP', action='store_true', help='whether to use CSPAttention, default=False', default=False)
    #parser.add_argument('--dilated', action='store_true', help='whether to use dilated causal convolution in encoder, default=False', default=False)
    parser.add_argument('--passthrough', action='store_true', help='whether to use passthrough mechanism in encoder, default=False', default=False)
    
    args = parser.parse_args()
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