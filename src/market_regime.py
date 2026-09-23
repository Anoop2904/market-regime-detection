import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
ASSETS=["Equity","GovtBond","Credit","Gold","Commodity","REIT"]
def generate_market(n_days=3900,seed=42):
 r=np.random.default_rng(seed); tr=np.array([[.965,.020,.005,.010],[.030,.945,.015,.010],[.015,.025,.925,.035],[.080,.015,.010,.895]]); z=np.zeros(n_days,dtype=int)
 for t in range(1,n_days): z[t]=r.choice(4,p=tr[z[t-1]])
 mu=np.array([[.00045,.00018,.00025,.00020,.00020,.00035],[.00005,.00012,.00005,.00030,.00028,.00000],[-.00110,.00035,-.00065,.00045,-.00050,-.00100],[.00065,.00010,.00040,.00015,.00030,.00055]])
 vol=np.array([[.010,.0035,.006,.008,.009,.009],[.012,.004,.008,.010,.012,.011],[.028,.006,.020,.016,.022,.025],[.014,.0038,.008,.009,.012,.012]])
 out=np.empty((n_days,6))
 for t in range(n_days): out[t]=mu[z[t]]+vol[z[t]]*(.70*r.normal(size=6)+.30*r.normal())
 idx=pd.bdate_range("2010-01-01",periods=n_days); return pd.DataFrame(out,index=idx,columns=ASSETS)
def make_features(x,w=21):
 f=pd.DataFrame(index=x.index); f["eq_return"]=x.Equity.rolling(w).mean(); f["eq_vol"]=x.Equity.rolling(w).std(); f["bond_return"]=x.GovtBond.rolling(w).mean(); f["credit_return"]=x.Credit.rolling(w).mean(); f["gold_return"]=x.Gold.rolling(w).mean(); f["dispersion"]=x.rolling(w).mean().std(axis=1); return f.dropna()
def metrics(x):
 ann=252; ar=(1+x).prod()**(ann/len(x))-1; av=x.std()*np.sqrt(ann); wealth=(1+x).cumprod(); dd=(wealth/wealth.cummax()-1).min(); return {"annual_return":ar,"annual_vol":av,"sharpe":ar/av,"max_drawdown":dd}
def run_experiment(seed=42):
 r=generate_market(seed=seed); f=make_features(r); s=int(.65*len(f)); train,test=f.iloc[:s],f.iloc[s:]; sc=StandardScaler(); xt=sc.fit_transform(train); models={}; bic={}
 for k in (2,3,4,5):
  m=GaussianMixture(n_components=k,covariance_type="full",n_init=10,random_state=42).fit(xt); models[k]=m; bic[k]=m.bic(xt)
 m=models[min(bic,key=bic.get)]; lab=m.predict(xt); tl=m.predict(sc.transform(test)); tr=r.loc[train.index].copy(); tr["regime"]=lab; risk=tr.groupby("regime")[ASSETS].std().mean(axis=1); hi,lo=risk.idxmax(),risk.idxmin(); w=np.full((len(test),6),1/6)
 for i,q in enumerate(tl): w[i]=[.08,.38,.08,.22,.12,.12] if q==hi else ([.38,.12,.18,.10,.08,.14] if q==lo else [.25,.20,.15,.15,.10,.15])
 rr=r.loc[test.index].iloc[1:].to_numpy(); adaptive=pd.Series((w[:-1]*rr).sum(axis=1),index=test.index[1:]); baseline=r.loc[test.index].iloc[1:].mean(axis=1); am,bm=metrics(adaptive),metrics(baseline)
 print("BIC:",{k:round(v,1) for k,v in bic.items()}); print("Selected regimes:",m.n_components); print("Adaptive:",{k:round(v,4) for k,v in am.items()}); print("Baseline:",{k:round(v,4) for k,v in bm.items()})
 return am,bm
