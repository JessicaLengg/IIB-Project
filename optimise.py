"""Freeze an own-trained model and FE-check bounded latent E22 searches."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from train import load_data,load_checkpoint
from models import hard_geometry,stiffness_matrix
from generate import connected
from fe import homogenize
from evaluate import save_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('model',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve completed/partial search.')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(20261108)
    data=load_data(args.dataset);vae,head,_=load_checkpoint(args.model)
    vae.requires_grad_(False);head.requires_grad_(False)
    weight_hash=hashlib.sha256((args.model/'model.pt').read_bytes()).hexdigest()
    train=np.flatnonzero((data['subset']=='train')&data['valid_FE'])
    e=1/np.linalg.inv(data['c_matrix'][train])[:,1,1];starts=[];seen=set()
    for i in train[np.argsort(-e)]:
        key=data['adjacency'][i].tobytes()
        if key not in seen:starts.append(int(i));seen.add(key)
        if len(starts)==8:break
    (args.output/'starts.json').write_text(json.dumps(starts)+'\n')
    previous=json.loads((args.model/'results.json').read_text())['reconstruction_FE_checks']
    fe_calls=0;cache={};physical=[];traces=[];summary=[];saved_candidates=[]

    def check_fe(probability,offsets,source,kind,step):
        nonlocal fe_calls
        a,p=hard_geometry(probability,offsets);key=hashlib.sha256(a.tobytes()+p.tobytes()).hexdigest()
        row={'source_id':source,'kind':kind,'step':step,'status':'failed','reason':'','E22':None,'cached':key in cache}
        if key in cache:row.update(cache[key])
        else:
            if fe_calls+previous>=64:raise RuntimeError('Prespecified 64 post-training FE budget reached')
            fe_calls+=1
            try:
                if not connected(a):raise ValueError('Decoded graph is not connected')
                edges=np.column_stack(np.where(np.triu(a,1)))
                if np.linalg.norm(p[edges[:,1]]-p[edges[:,0]],axis=1).min()<1e-8:raise ValueError('Degenerate decoded beam')
                c,info=homogenize(a,p,weld_intersections=False)
                if info['minimum_stiffness_eigenvalue']<=1e-10:raise ValueError('Non-positive decoded FE stiffness')
                row.update(status='passed',E22=float(1/np.linalg.inv(info['c_matrix'])[1,1]))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:row['reason']=str(error)
            cache[key]={k:row[k] for k in ['status','reason','E22']}
        physical.append(row);save_csv(args.output/'FE_checks.csv',physical)
        saved_candidates.append({'source':source,'kind':kind,'a':a,'p':p,'offsets':offsets.copy()})
        return row

    with threadpool_limits(limits=2):
        for i in starts:
            with torch.no_grad():
                _,mu,_=vae.encoder(data['a'][i:i+1],data['x'][i:i+1]);pa,px=vae.decoder(mu)
            baseline=check_fe(pa.numpy()[0],px.numpy()[0],i,'decoded_start',0)
            z0=mu.detach();z=z0.clone().requires_grad_(True);optimizer=torch.optim.Adam([z],lr=1e-3)
            best_z=z0.clone();best_score=-np.inf;best_step=0;reason='budget_completed';valid_steps=0
            for step in range(401):
                a,x=vae.decoder(z);_,encoded,_=vae.encoder(a,x);c=stiffness_matrix(head(encoded))
                if not torch.isfinite(c).all() or torch.linalg.eigvalsh(c.detach())[0,0]<=1e-8:
                    reason='non_positive_surrogate_C';break
                e22=1/torch.linalg.inv(c)[0,1,1]
                if not torch.isfinite(e22) or e22<=0:reason='invalid_surrogate_E22';break
                value=float(e22.detach());valid_steps+=1
                traces.append({'source_id':i,'step':step,'predicted_E22':value})
                if value>best_score:best_score=value;best_step=step;best_z=z.detach().clone()
                if step==200:
                    check_fe(a.detach().numpy()[0],x.detach().numpy()[0],i,'midpoint',step)
                if step==400:break
                loss=((e22-.15)/.15).square();optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():z.copy_(torch.maximum(torch.minimum(z,z0+.5),z0-.5))
            with torch.no_grad():
                a,x=vae.decoder(best_z)
                endpoint=check_fe(a.numpy()[0],x.numpy()[0],i,'surrogate_selected',best_step)
                displacement=np.random.default_rng(20261108+i).uniform(-.5,.5,(1,48))
                a,x=vae.decoder(z0+torch.tensor(displacement,dtype=torch.float32))
                random=check_fe(a.numpy()[0],x.numpy()[0],i,'one_random_displacement',0)
            original=float(1/np.linalg.inv(data['c_matrix'][i])[1,1])
            actual=endpoint['E22'];initial=baseline['E22']
            row={'source_id':i,'original_FE_E22':original,'decoded_start_FE_E22':initial,
                 'endpoint_FE_E22':actual,'random_FE_E22':random['E22'],'selected_step':best_step,
                 'valid_surrogate_evaluations':valid_steps,'stop_reason':reason,'endpoint_status':endpoint['status'],
                 'gain_vs_original_percent':100*(actual/original-1) if actual is not None else None,
                 'gain_vs_decoded_start_percent':100*(actual/initial-1) if actual is not None and initial else None}
            summary.append(row);save_csv(args.output/'comparison.csv',summary)
            np.savez_compressed(args.output/f'latent_{i}.npz',start=z0.numpy(),selected=best_z.numpy())
            print(row,flush=True)
    save_csv(args.output/'trajectory.csv',traces) if traces else None
    np.savez_compressed(args.output/'candidates.npz',source_id=[v['source'] for v in saved_candidates],
                        kind=[v['kind'] for v in saved_candidates],adjacency=[v['a'] for v in saved_candidates],
                        positions=[v['p'] for v in saved_candidates],offsets=[v['offsets'] for v in saved_candidates])
    eligible=[r for r in summary if r['endpoint_status']=='passed' and r['selected_step']>0]
    improved=[r for r in eligible if r['gain_vs_original_percent']>0]
    assert hashlib.sha256((args.model/'model.pt').read_bytes()).hexdigest()==weight_hash
    result={'status':'complete','starts':len(starts),'valid_optimised_endpoints':len(eligible),'improved_vs_original':len(improved),
            'new_FE_attempts':fe_calls,'earlier_reconstruction_FE_attempts':previous,'post_training_FE_cap':64,
            'best_endpoint':max(eligible,key=lambda r:r['endpoint_FE_E22']) if eligible else None,
            'own_model_sha256':weight_hash,'weights_unchanged':True,'stress_constraint_used':False,
            'scope':'Own-trained frozen model; training-source starts; bounded local search, not a global optimum or speed benchmark.'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Own-model latent stiffness search','',
           'Freeze the VAE and stiffness head trained from scratch on our own FE-labelled structures. Eight training-source topologies initialize the fixed bounded searches. No test examples choose starts, trajectories or settings.','',
           '| Source ID | Original FE E22 | Decoded-start FE E22 | Selected-endpoint FE E22 | Gain versus original (%) | Selected step | Stop reason |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary:
        lines.append(f'| {r["source_id"]} | {r["original_FE_E22"]:.6g} | {r["decoded_start_FE_E22"]} | {r["endpoint_FE_E22"]} | {r["gain_vs_original_percent"]} | {r["selected_step"]} | {r["stop_reason"]} |')
    lines+=['',f'Endpoints after at least one latent update: {len(eligible)}/{len(starts)}. Those exceeding their own original FE source: {len(improved)}/{len(starts)}. Step-zero fallbacks are retained but are not optimised results.',
            '',f'New post-training search FE attempts: {fe_calls}; earlier reconstructed-test FE attempts: {previous}; combined cap: 64. The full dataset FE labelling is a separate upfront cost. Exact repeated decoded geometries reuse the search cache.',
            '', 'The endpoint is selected by surrogate prediction before its FE check. Midpoint checks do not update the optimisation path. Both gains versus the original structure and versus the decoded start are saved; recovering a starting reconstruction loss is not necessarily a better design than the original.',
            '', 'The random displacement is one fixed diagnostic per source, not an equally tuned optimisation benchmark. No faster-convergence, global-optimality, novel-topology, stress-limit or manufacturing claim is made. Invalid predicted stiffness and actual FE failures remain visible. Negative results are valid outcomes. The saved own-trained checkpoint hash is unchanged.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');k=np.arange(len(summary));width=.25
    for shift,key,title in [(-1,'original_FE_E22','Original source'),(0,'decoded_start_FE_E22','Decoded start'),(1,'endpoint_FE_E22','Surrogate-selected endpoint')]:
        ax.bar(k+shift*width,[r[key] if r[key] is not None else np.nan for r in summary],width,label=title)
    ax.set(xticks=k,xticklabels=[str(r['source_id']) for r in summary],xlabel='Own training source ID',ylabel='Actual FE E22 / Es')
    ax.legend(fontsize=8);fig.savefig(args.output/'search_results.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
