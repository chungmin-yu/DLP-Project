import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize(args):
	vali_preds = np.load(f"{args.model_path + args.path}/vali_predictions.npy")
	vali_trues = np.load(f"{args.model_path + args.path}/vali_truths.npy")
	preds = np.load(f"{args.model_path + args.path}/predictions.npy")
	trues = np.load(f"{args.model_path + args.path}/truths.npy")
	
	vali_preds = (vali_preds * args.std) + args.mean
	vali_trues = (vali_trues * args.std) + args.mean
	preds = (preds * args.std) + args.mean
	trues = (trues * args.std) + args.mean

	vali_preds = vali_preds[:,:,-1]
	vali_trues = vali_trues[:,:,-1]
	vali_size = np.size(vali_trues)
	preds = preds[:,:,-1]
	trues = trues[:,:,-1]
	size = np.size(trues)

	plt.figure(figsize=(12, 10))
	plt.subplot(211)
	plt.title('Validate Data', fontsize=18)
	plt.xlabel('date', fontsize=12) 
	plt.ylabel('closing return', fontsize=12) 
	plt.plot(vali_trues.reshape(vali_size), linewidth=3, label='GroundTruth')
	plt.plot(vali_preds.reshape(vali_size), linewidth=3, label='Prediction')
	plt.legend()
	plt.subplot(212)
	plt.title('Test Data', fontsize=18)
	plt.xlabel('date', fontsize=12) 
	plt.ylabel('closing return', fontsize=12) 
	plt.plot(trues.reshape(size), linewidth=3, label='GroundTruth')
	plt.plot(preds.reshape(size), linewidth=3, label='Prediction')
	plt.legend()
	plt.tight_layout()
	plt.show()

def predict(args):
	preds = np.load(f"{args.model_path + args.path}/pred_predictions.npy")
	trues = np.load(f"{args.model_path + args.path}/pred_truths.npy")

	preds = (preds * args.std) + args.mean
	trues = (trues * args.std) + args.mean

	preds = preds.flatten()
	trues = trues.flatten()
	trues_date=[i for i in range(len(trues))]
	preds_date=[i for i in range(len(trues)-1, len(trues)-6, -1)]

	plt.figure()
	plt.title('Prediction Data', fontsize=18)
	plt.xlabel('date', fontsize=12) 
	plt.ylabel('closing return', fontsize=12) 
	plt.plot(trues_date, trues, linewidth=3, label='GroundTruth')
	plt.plot(preds_date, preds, linewidth=3, label='Prediction')
	plt.legend()
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_attention', action='store_true')
	parser.add_argument('--model_path', type=str, default='results/', help='root path of the model result')
	parser.add_argument('--path', type=str, default='', help='path of the model path')
	parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
	parser.add_argument('--distil', action='store_true', help='whether to use distilling in encoder', default=False)
	parser.add_argument('--passthrough', action='store_true', help='whether to use passthrough mechanism in encoder, default=False', default=False)
	parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')
	
	parser.add_argument('--mode', type=int, default=1, help='visualize validation & test data : 1 / predict data : 0')
	parser.add_argument('--mean', type=float, default=1, help='mean')
	parser.add_argument('--std', type=float, default=1, help='std')
	args = parser.parse_args()
	
	if args.mode:
		visualize(args)
	else:
		predict(args)