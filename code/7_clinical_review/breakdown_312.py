import sys, numpy as np, pandas as pd
U=sys.argv[1]
def norm(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().lower(); return "Partially" if s.startswith("partial") else {"yes":"Yes","no":"No"}.get(s,np.nan)
def load(f):
    d=pd.read_excel(f"{U}/{f}",sheet_name="Reviews"); d=d[d["Matched Event ID"].notna()].copy()
    d.index=d["Event set"].astype(str)+"|"+d["Matched Event ID"].astype(str); return d
A,Z=load("527a7308-physician_1_312.xlsx"),load("007b2db9-physician_2_312.xlsx")
d=pd.DataFrame({"a":A["Clinical meaning changed?"].map(norm),"z":Z["Clinical meaning changed?"].reindex(A.index).map(norm),
 "corpus":A["Corpus"].fillna("Fareez"),"cue":A["Cue type"].astype(str),"spk":A["Speaker"]})
d["yes"]=(d.a=="Yes")&(d.z=="Yes")
CATS=["No","Partially","Yes"]
def kappa(x,y):
    po=np.mean(x==y); pe=sum(np.mean(x==k)*np.mean(y==k) for k in CATS); return (po-pe)/(1-pe), po
print("n", len(d), "agreed yes", d.yes.sum())
for c,g in d.groupby("corpus"):
    k,po=kappa(g.a.values,g.z.values); print(f"{c}: n={len(g)} kappa={k:.3f} agree={po:.3f} yes={g.yes.sum()}")
print(d.groupby("cue").agg(n=("yes","size"),yes=("yes","sum")).to_string())
print(d.groupby("spk").agg(n=("yes","size"),yes=("yes","sum")).to_string())
