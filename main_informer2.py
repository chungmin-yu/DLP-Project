''' Libraries '''
import os
import time
import argparse

import torch
from exp.exp_informer2 import Exp_Informer
#from memory_profiler import profile
import os, psutil
import GPUtil as GPU

''' Parameters '''
CONTINUE_TRAIN = False
MODEL_DIRECTORY = "2021-03-18_04.37.27_bs25_sl400_ll300_pl23_sFalse_do0.05_lr0.0001_lradjunknow_fc5_dtTrue_1"


''' Execution '''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--data', type=str, default='custom', help='data')
    parser.add_argument('--root_path', type=str, default='./data/stocks_10/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='AIRT.csv', help='data file')
    parser.add_argument('--data_path2', type=str, default='AIRT.csv', help='data file')
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
    parser.add_argument('--beta', type=float, default=0.1, help='market index context aggregation')
    #parser.add_argument('--CSP', action='store_true', help='whether to use CSPAttention, default=False', default=False)
    #parser.add_argument('--dilated', action='store_true', help='whether to use dilated causal convolution in encoder, default=False', default=False)
    parser.add_argument('--passthrough', action='store_true', help='whether to use passthrough mechanism in encoder, default=False', default=False)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() else False

    print('Args in experiment:')
    print(args)

    Exp = Exp_Informer
    torch.cuda.empty_cache()

    now = time.localtime()
    now_date = time.strftime('%Y-%m-%d', now)
    now_time = time.strftime('%H.%M.%S', now)
    start = time.time()

    for ii in range(args.itr):
        # setting record of experiments
        # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model, args.data, args.features, 
        #             args.seq_len, args.label_len, args.pred_len,
        #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.des, ii)
        setting = f"{now_date}_{now_time}_bs{args.batch_size}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_fn{args.data_path}_lr{args.learning_rate}_lradj{args.lradj}_{ii+1}"
        
        args.ii = ii + 1
        exp = Exp(args) # set experiments

        if CONTINUE_TRAIN:
            exp.model.load_state_dict(torch.load(f"results/{MODEL_DIRECTORY}/checkpoint.pth"))
            exp.model.train()

        print(f">>>>>>> training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)
        
        print(f">>>>>>> testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)

        # Memory usage 
        process = psutil.Process(os.getpid())
        print("CPU Memory usage(MB): ", process.memory_info().rss/ 1024 ** 2, " MB")
        gpu = GPU.getGPUs()[0]
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
        end = time.time()
        print("Excution time：%f secs" % (end - start))

        torch.cuda.empty_cache()