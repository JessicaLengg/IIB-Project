"""Train one own-data joint VAE/predictor from random weights."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from models import fresh_model, PropertyHead, UPPER, EDGES, stiffness_matrix


def load_data(folder):
    with np.load(folder/'structures.npz') as source: data={k:source[k].copy() for k in source.files}
    with np.load(folder/'labels.npz') as source:
        assert np.array_equal(data['sample_ids'],source['sample_ids'])
        data.update({k:source[k].copy() for k in source.files})
    with np.load(folder/'splits.npz') as source:
        assert np.array_equal(data['sample_ids'],source['sample_ids']); data['subset']=source['subset'].copy()
    data['a']=torch.tensor(data['adjacency'][:,UPPER[0],UPPER[1]],dtype=torch.float32)
    data['x']=torch.tensor(data['offsets'],dtype=torch.float32)
    return data


@torch.no_grad()
def predict(vae,head,data,ids,sampled=False,seed=73519):
    vae.eval(); head.eval(); result={k:[] for k in ['adjacency_probability','offset_prediction','stiffness_prediction','mean']}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for start in range(0,len(ids),128):
            take=ids[start:start+128];z,mu,_=vae.encoder(data['a'][take],data['x'][take]);a,x=vae.decoder(z if sampled else mu)
            for key,value in zip(result,[a,x,head(mu),mu]):result[key].append(value.numpy())
    return {key:np.concatenate(values) for key,values in result.items()}


def scores(pred,data,ids,std):
    off=UPPER[0]!=UPPER[1]
    wrong=np.count_nonzero((pred['adjacency_probability'][:,off]>=.5)!=(data['a'][ids].numpy()[:,off]>=.5),axis=1)
    diagonal_errors=np.count_nonzero(pred['adjacency_probability'][:,~off]>=.5)
    c=pred['stiffness_prediction'];truth=data['stiffness'][ids]
    return {'n':len(ids),'exact_topologies':int((wrong==0).sum()),'exact_fraction':float(np.mean(wrong==0)),
            'mean_wrong_edges':float(wrong.mean()),'matrix_accuracy':float(1-(2*wrong.sum()+diagonal_errors)/(len(ids)*729)),
            'offset_RMSE':float(np.sqrt(np.mean((pred['offset_prediction']-data['offsets'][ids])**2))),
            'C_RMSE':float(np.sqrt(np.mean((c-truth)**2))),
            'C_normalized_MSE':float(np.mean(((c-truth)/std)**2)),
            'valid_predicted_C':int((np.linalg.eigvalsh(stiffness_matrix(c))[:,0]>0).sum())}


def load_checkpoint(folder):
    saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    vae=fresh_model();head=PropertyHead(saved['label_mean'],saved['label_std'])
    vae.load_state_dict(saved['vae']);head.load_state_dict(saved['head'])
    return vae.eval(),head.eval(),saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=200);parser.add_argument('--seed',type=int,default=20261108)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve completed/partial fit; use a fresh output folder.')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(args.seed)
    data=load_data(args.dataset)
    ids={s:np.flatnonzero((data['subset']==s)&data['valid_FE']) for s in ['train','validation','test']}
    assert all(len(i) for i in ids.values())
    # Prespecify mechanical reconstruction checks without inspecting test outcomes.
    fixed=np.sort(np.random.default_rng(args.seed).choice(ids['test'],12,replace=False))
    (args.output/'fixed_FE_ids.json').write_text(json.dumps(fixed.tolist())+'\n')
    truth=data['stiffness'].astype(np.float32);mean=truth[ids['train']].mean(0);std=np.maximum(truth[ids['train']].std(0),1e-7)
    np.savez_compressed(args.output/'normalization.npz',mean=mean,std=std)
    vae=fresh_model();head=PropertyHead(mean,std)
    optimizer=torch.optim.Adam(list(vae.parameters())+list(head.parameters()),lr=3e-4)
    target=torch.tensor(truth);history=[];best=None;started=time.perf_counter()
    batch_rng=torch.Generator().manual_seed(args.seed)
    for epoch in range(1,args.epochs+1):
        if epoch==31:
            for group in optimizer.param_groups:group['lr']=1e-4
        beta=1e-4*min(max(epoch-30,0)/50,1);vae.train();head.train();total=0.
        order=torch.tensor(ids['train'])[torch.randperm(len(ids['train']),generator=batch_rng)]
        for batch in order.split(128):
            z,mu,lv=vae.encoder(data['a'][batch],data['x'][batch]);a,x=vae.decoder(mu if epoch<=30 else z)
            kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
            loss=F.binary_cross_entropy(a,data['a'][batch])+10*(x-data['x'][batch]).square().mean()
            loss=loss+.1*((head(mu)-target[batch])/head.label_std).square().mean()+beta*kl
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vae.parameters())+list(head.parameters()),10);optimizer.step()
            total+=float(loss.detach())*len(batch)
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            metric=scores(predict(vae,head,data,ids['validation']),data,ids['validation'],std)
            row={'epoch':epoch,'training_loss':total/len(order),**metric};history.append(row)
            with (args.output/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=row,lineterminator="\n");writer.writeheader();writer.writerows(history)
            key=(-metric['exact_fraction'],metric['mean_wrong_edges'],metric['offset_RMSE'],metric['C_normalized_MSE'])
            if best is None or key<best:
                best=key
                torch.save({'vae':vae.state_dict(),'head':head.state_dict(),'epoch':epoch,'seed':args.seed,
                            'label_mean':mean.tolist(),'label_std':std.tolist()},args.output/'model.pt')
            print(f'Epoch {epoch}: validation exact {metric["exact_topologies"]}/{metric["n"]}; offset RMSE {metric["offset_RMSE"]:.5f}',flush=True)
    meta={'seed':args.seed,'epochs':args.epochs,'samples':{s:len(i) for s,i in ids.items()},'training_seconds':time.perf_counter()-started,
          'pretrained_weights_used':False,'dataset_sha256':hashlib.sha256((args.dataset/'structures.npz').read_bytes()).hexdigest(),
          'labels_sha256':hashlib.sha256((args.dataset/'labels.npz').read_bytes()).hexdigest(),
          'test_selection':'not used for checkpoint/hyperparameter choice','architecture':'upstream VAE, own 48-128-128-9 head',
          'sampler':'Original shared Gaussian scalar per sample; exp(logvar) scale and original KL retained.',
          'checkpoint_selection':'validation exact recovery, wrong edges, offsets, normalized property error'}
    (args.output/'training.json').write_text(json.dumps(meta,indent=2)+'\n')
    from evaluate import evaluate
    evaluate(args.dataset,args.output)


if __name__=='__main__':main()
