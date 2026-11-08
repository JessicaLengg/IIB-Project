"""Evaluate selected own-trained weights after validation-only selection."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import EDGES, COMPONENTS, hard_geometry, stiffness_matrix
from train import load_data, load_checkpoint, predict, scores
from fe import homogenize


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator="\n");writer.writeheader();writer.writerows(rows)


def evaluate(dataset,output):
    torch.set_num_threads(2);data=load_data(dataset);vae,head,saved=load_checkpoint(output)
    std=np.asarray(saved['label_std']);rows=[];properties=[];details=[]
    ids=np.flatnonzero(data['valid_FE']);predictions={}
    for mode,sampled in [('mean',False),('native_sample',True)]:
        pred=predict(vae,head,data,ids,sampled);predictions[mode]=pred
        np.savez_compressed(output/f'{mode}_predictions.npz',sample_ids=ids,**pred)
        for subset in ['train','validation','test']:
            take=data['subset'][ids]==subset;part=ids[take];local={k:v[take] for k,v in pred.items()}
            rows.append({'subset':subset,'mode':mode,**scores(local,data,part,std)})
            if mode=='mean':
                for j,name in enumerate(COMPONENTS):
                    y=data['stiffness'][part,j];p=local['stiffness_prediction'][:,j]
                    properties.append({'subset':subset,'component':name,'RMSE':float(np.sqrt(np.mean((p-y)**2))),
                                       'R2':float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))})
                for j,i in enumerate(part):
                    a,p=hard_geometry(local['adjacency_probability'][j],local['offset_prediction'][j]);active=data['adjacency'][i].sum(0)>0
                    delta=abs(p[active]-data['positions'][i][active])
                    details.append({'sample_id':int(i),'subset':subset,'wrong_edges':int(np.count_nonzero(a[EDGES]!=data['adjacency'][i][EDGES])),
                                    'active_coordinate_MAE':float(delta.mean()),'maximum_coordinate_error':float(delta.max())})
    save_csv(output/'metrics.csv',rows);save_csv(output/'property_metrics.csv',properties);save_csv(output/'geometry_metrics.csv',details)
    physical=[];pred=predictions['mean'];lookup={int(i):j for j,i in enumerate(ids)}
    fixed=json.loads((output/'fixed_FE_ids.json').read_text())
    for i in fixed:
        j=lookup[i];a,p=hard_geometry(pred['adjacency_probability'][j],pred['offset_prediction'][j])
        row={'sample_id':i,'status':'failed','reason':'','original_E22':float(1/np.linalg.inv(data['c_matrix'][i])[1,1]),
             'reconstructed_E22':None,'E22_change_percent':None,'exact_topology':bool(np.array_equal(a[EDGES],data['adjacency'][i][EDGES]))}
        try:
            c,info=homogenize(a,p,weld_intersections=False)
            if info['minimum_stiffness_eigenvalue']<=1e-10:raise ValueError('Non-positive reconstructed FE stiffness')
            e=1/np.linalg.inv(info['c_matrix'])[1,1]
            row.update(status='passed',reconstructed_E22=float(e),E22_change_percent=float(100*(e/row['original_E22']-1)))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:row['reason']=str(error)
        physical.append(row)
    save_csv(output/'reconstruction_FE.csv',physical)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    mean=[r for r in rows if r['mode']=='mean']
    axes[0].bar([r['subset'] for r in mean],[100*r['exact_fraction'] for r in mean],color='#256482')
    axes[0].set(ylabel='Complete topology recovery (%)',ylim=(0,105))
    test=data['subset'][ids]=='test';truth=data['stiffness'][ids[test],3];p=pred['stiffness_prediction'][test,3]
    axes[1].scatter(truth,p,s=8,alpha=.5);bounds=[min(truth.min(),p.min()),max(truth.max(),p.max())]
    axes[1].plot(bounds,bounds,'k--');axes[1].set(xlabel='FE C22',ylabel='Predicted C22',title='Test property prediction')
    fig.savefig(output/'training_results.png',dpi=170);plt.close(fig)
    test_ids=ids[test]
    topology=lambda part:{data['adjacency'][i].tobytes() for i in part}
    train_topology=topology(ids[data['subset'][ids]=='train'])
    result={'status':'complete','selected_epoch':saved['epoch'],'metrics':rows,'reconstruction_FE_checks':len(physical),
            'reconstruction_FE_passed':sum(r['status']=='passed' for r in physical),'independent_test_used_for_selection':False,
            'test_topologies':len(topology(test_ids)),
            'test_rows_with_topology_in_training':sum(data['adjacency'][i].tobytes() in train_topology for i in test_ids)}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Own-data reconstruction and stiffness prediction','',
           f'One scratch joint fit; selected epoch {saved["epoch"]}. No author checkpoint is read. The upstream VAE architecture and native sampler are retained, with our small stiffness head and loss.','',
           '| Partition | Mode | Exact topology | Matrix accuracy | Offset RMSE | Nine-C RMSE | Positive-definite predicted C |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f'| {r["subset"]} | {r["mode"]} | {r["exact_topologies"]}/{r["n"]} | {100*r["matrix_accuracy"]:.4f}% | {r["offset_RMSE"]:.5g} | {r["C_RMSE"]:.5g} | {r["valid_predicted_C"]}/{r["n"]} |')
    geo=[r for r in details if r['subset']=='test'];passed=[r for r in physical if r['status']=='passed']
    lines+=['',f'Test contains {result["test_topologies"]} distinct topologies; {result["test_rows_with_topology_in_training"]}/{len(test_ids)} test rows share their topology with training. This is random sample splitting, not a topology-disjoint benchmark. Test did not select checkpoints or hyperparameters.',
            '',f'Test median active coordinate MAE: {np.median([r["active_coordinate_MAE"] for r in geo]):.6g}; median maximum active coordinate component error: {np.median([r["maximum_coordinate_error"] for r in geo]):.6g}, relative to octant edge length 1. Exact connections do not imply exact positions.',
            '',f'Prespecified reconstructed-geometry FE checks: {len(passed)}/{len(physical)} passed. Median absolute E22 change among successful solves: {np.median([abs(r["E22_change_percent"]) for r in passed]) if passed else "undefined"}%. These twelve checks are not all-test FE verification.',
            '', 'Property R2/RMSE by component are in property_metrics.csv. One dataset and training seed do not establish broad generative validity. Native VAE sampling variation is not calibrated property uncertainty. FE stability is not manufacturing or strength validation. See EXPERIMENT.md for the fixed recipe.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();evaluate(args.dataset,args.output)
