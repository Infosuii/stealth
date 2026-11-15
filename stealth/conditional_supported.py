"""Fixed effective-support constraint on otherwise unchanged conditional covariance weights."""
import numpy as np
from stealth.conditional_blend import solve
from stealth.online_missing import utility


def constrained_mass(age,kernel,maximum,fraction,cap):
    age,kernel=np.asarray(age,float),np.asarray(kernel,float)
    target=min(float(cap),float(fraction)/(age@age))
    delta=kernel-age;alpha=float(maximum)
    mass=(1-alpha)*age+alpha*kernel
    if 1/(mass@mass)<target:
        A=delta@delta;B=2*(age@delta);C=age@age-1/target
        root=(-B+np.sqrt(max(B*B-4*A*C,0)))/(2*A)
        alpha=min(alpha,max(0.,float(root)))
        mass=(1-alpha)*age+alpha*kernel
    return mass,alpha,target


def predict(P,y,era,core,states,cfg,conditional):
    P,y,era,core=np.asarray(P,float),np.asarray(y,float),np.asarray(era),np.asarray(core,float)
    states=np.asarray(states,float);unique=np.unique(era);groups=[np.flatnonzero(era==e) for e in unique]
    if len(states)!=len(unique) or not np.isfinite(states).all() or not np.isfinite(P).all():raise ValueError('invalid state/prediction layout')
    prior=np.asarray(cfg['prior'],float);d=P.shape[1]
    if len(prior)!=d or np.min(prior)<0 or not np.isclose(prior.sum(),1.):raise ValueError('invalid prior')
    C=np.empty((len(unique),d,d));mu=np.empty((len(unique),d));next_matured=0
    output=np.full(len(y),np.nan);trace=[]
    for j,(e,ids) in enumerate(zip(unique,groups)):
        cut=e-cfg['gap']
        while next_matured<len(unique) and unique[next_matured]<cut:
            k=next_matured;m=groups[k];z=P[m]-P[m].mean(axis=0);t=utility(y[m],era[m],P[m,0],core[m])
            C[k]=z.T@z/len(m);mu[k]=z.T@t/len(m);next_matured+=1
        keep=np.flatnonzero(unique[:next_matured]>=cut-cfg['history'])
        w=prior.copy();ess=0.;distance=None;alpha=0.;floor=0.
        if len(keep)>=cfg['min_eras']:
            age=np.exp2((unique[keep]-unique[keep].max())/cfg['half_life']);age/=age.sum();mass=age
            if conditional:
                center=age@states[keep];sd=np.maximum(np.sqrt(age@((states[keep]-center)**2)),1e-6)
                distances=np.mean(((states[keep]-states[j])/sd)**2,axis=1)
                log_kernel=-distances/(2*cfg['bandwidth']**2);log_kernel-=log_kernel.max()
                kernel=age*np.exp(log_kernel);kernel/=kernel.sum()
                mass,alpha,floor=constrained_mass(age,kernel,cfg['context_mass'],cfg['support_fraction'],cfg['support_cap'])
                distance=float(distances.min())
            ess=float(1/(mass@mass));w=solve(np.einsum('i,ijk->jk',mass,C[keep]),mass@mu[keep],prior,cfg['ridge'])
        w=(1-cfg['anchor'])*w;w[0]+=cfg['anchor'];output[ids]=P[ids]@w
        trace.append(dict(era=int(e),train_max=int(unique[next_matured-1]) if next_matured else None,
                          fitting_eras=len(keep),effective_eras=ess,nearest_state_distance=distance,weights=w.tolist(),
                          applied_context_mass=alpha,required_effective_eras=floor))
    if not np.isfinite(output).all():raise RuntimeError('uncovered daily predictions')
    return output,trace
