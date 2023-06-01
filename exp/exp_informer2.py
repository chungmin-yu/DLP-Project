from data.data_loader2 import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model2 import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')


# Add by Stock
from tqdm import tqdm
from plot_history import plot_figure


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                # self.args.CSP,
                # self.args.dilated,
                self.args.passthrough,
                self.args.beta,
                self.device
            )
        
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1' : Dataset_ETT_hour,
            'ETTh2' : Dataset_ETT_hour,
            'ETTm1' : Dataset_ETT_minute,
            'ETTm2' : Dataset_ETT_minute,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 1 if args.embed == 'timeF' else 0

        if flag == 'test' or flag == 'predict':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            data_path2=args.data_path2,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            batch_size=args.batch_size,  # Add by Stock
            scale=args.scale,            # Add by Stock
            timeenc=timeenc,
            freq=args.freq
        )
        print("{0} data length: {1}".format(flag, len(data_set)))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, scaler, setting):
        self.model.eval()

        scaled_total_loss = []
        # real_total_loss = []

        preds = []
        trues = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x2, batch_y2, batch_x_mark2, batch_y_mark2) in enumerate(vali_loader):
            
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # context aggregation
            batch_x2 = batch_x2.double().to(self.device)
            batch_y2 = batch_y2.double()
            batch_x_mark2 = batch_x_mark2.double().to(self.device)
            batch_y_mark2 = batch_y_mark2.double().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)

            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            scaled_loss = criterion(pred, true)
            scaled_total_loss.append(scaled_loss)

            # real_loss = criterion(scaler.inverse_transform(pred), scaler.inverse_transform(true))
            # real_total_loss.append(real_loss)

            preds.append(pred.numpy())
            trues.append(true.numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        #print('vali shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('vali shape:', preds.shape, trues.shape)

        if setting is not None:
            # result save
            saving_directory = f"results/{setting}/"
            np.save(f"{saving_directory}/vali_predictions.npy", preds)
            np.save(f"{saving_directory}/vali_truths.npy", trues)
            
        scaled_loss = np.average(scaled_total_loss)
        # real_loss = np.average(real_total_loss)

        self.model.train()
        # return scaled_loss, real_loss
        return scaled_loss, None
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')
        print("")

        path = f"results/{setting}"
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        history = {
            'scaled': { 'train': [], 'vali': [], 'test': [] },
            # 'real'  : { 'train': [], 'vali': [], 'test': [] },
            'learning_rate': []
        }

        for epoch in range(self.args.train_epochs):
            scaled_total_loss = []
            # real_total_loss = []

            # backward four is market index aggregation
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x2, batch_y2, batch_x_mark2, batch_y_mark2) in enumerate(tqdm(train_loader, desc=f"Iteration: {self.args.ii:>2}, Epoch: {epoch+1:>2}", ascii=True)):
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double()
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                # context aggregation
                batch_x2 = batch_x2.double().to(self.device)
                batch_y2 = batch_y2.double()
                batch_x_mark2 = batch_x_mark2.double().to(self.device)
                batch_y_mark2 = batch_y_mark2.double().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
                
                scaled_loss = criterion(outputs, batch_y)
                scaled_total_loss.append(scaled_loss.item())

                # test = train_data.scaler.inverse_transform(outputs.detach().cpu())
                # print(test[:, :, -2:-1])
                # real_loss = criterion(test, train_data.scaler.inverse_transform(batch_y.detach().cpu()))
                # real_total_loss.append(real_loss.item())
                
                scaled_loss.backward()
                # real_loss.backward()
                model_optim.step()

            train_scaled_loss = np.average(scaled_total_loss)
            # train_real_loss = np.average(real_total_loss)
            vali_scaled_loss, vali_real_loss = self.vali(vali_data, vali_loader, criterion, train_data.scaler, setting)
            test_scaled_loss, test_real_loss = self.vali(test_data, test_loader, criterion, train_data.scaler, None)

            history['scaled']['train'].append(train_scaled_loss)
            history['scaled']['vali'].append(vali_scaled_loss)
            history['scaled']['test'].append(test_scaled_loss)

            # history['real']['train'].append(train_real_loss)
            # history['real']['vali'].append(vali_real_loss)
            # history['real']['test'].append(test_real_loss)

            print(f"Train Loss: {train_scaled_loss:.5f}, Vali Loss: {vali_scaled_loss:.5f}, Test Loss: {test_scaled_loss:.5f},")
            # print(f"Train Scaled Loss: {train_scaled_loss:.5f}, Train Real Loss: {train_real_loss:.5f} | \
            #     Vali Scaled Loss: {vali_scaled_loss:.5f}, Vali Real Loss: {vali_real_loss:.5f} | \
            #     Test Scaled Loss: {test_scaled_loss:.5f}, Test Real Loss: {test_real_loss:.5f},")
            
            early_stopping(vali_scaled_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping\n")
                break
            
            learning_rate = adjust_learning_rate(model_optim, epoch+1, self.args)
            history['learning_rate'].append(learning_rate)

            plot_figure(path, history, self.args)

        best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y2, batch_x_mark2, batch_y_mark2) in enumerate(test_loader):
            
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # context aggregation
            batch_x2 = batch_x2.double().to(self.device)
            batch_y2 = batch_y2.double()
            batch_x_mark2 = batch_x_mark2.double().to(self.device)
            batch_y_mark2 = batch_y_mark2.double().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)
            
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        saving_directory = f"results/{setting}/"

        mse, mae, rmse, mape, mspe = metric(preds, trues)
        print('mse: {}, mae: {}, rmse: {}, mape: {}, mspe: {}'.format(mse, mae, rmse, mape, mspe))
        # print('MSE: {} | MAE: {}'.format(mse, mae))

        np.save(f"{saving_directory}/metrics.npy", np.array([mse, mae, rmse, mape, mspe]))
        np.save(f"{saving_directory}/predictions.npy", preds)
        np.save(f"{saving_directory}/truths.npy", trues)

        return


    # Add by Stock
    def predict(self, saving_directory):

        preds = []
        trues = []
        
        predict_data, predict_loader = self._get_data(flag='predict')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y2, batch_x_mark2, batch_y_mark2) in enumerate(predict_loader):
            
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double().to(self.device)
            batch_y_mark = batch_y_mark.double().to(self.device)

            # context aggregation
            batch_x2 = batch_x2.double().to(self.device)
            batch_y2 = batch_y2.double()
            batch_x_mark2 = batch_x_mark2.double().to(self.device)
            batch_y_mark2 = batch_y_mark2.double().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x2, batch_x_mark2)

            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        # print('predict shape:', preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('predict shape:', preds.shape)

        print("prediction:")
        print(preds[-1])
        print("groundtruth")
        print(trues[-1])
        # predict save
        np.save(f"{saving_directory}/last_prediction.npy", preds[-1])
        return