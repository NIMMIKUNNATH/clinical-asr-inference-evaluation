import pandas as pd
S="/mnt/d/fareez_optuna/final/scores/"
for cfg in ["defaults","tuned"]:
    d=pd.read_csv(S+cfg+".csv")
    N=d.sa_n.sum(); assert (d.sa_n==d.ref_words).all() or True
    sa=100*d.sa_err.sum()/N; doc=100*d.doc_err.sum()/d.doc_n.sum(); pat=100*d.pat_err.sum()/d.pat_n.sum()
    roles=100*(d.doc_err.sum()+d.pat_err.sum())/N; un=100*d.unassigned_words.sum()/N; cp=100*d.cp_err.sum()/d.cp_n.sum()
    fsd=d.first_speaker_is_doctor
    ok=d[fsd==True]; bad=d[fsd==False]
    print(f"== {cfg}: n={len(d)}  SA-WER {sa:.2f}  cpWER {cp:.2f}  clinician {doc:.2f}  patient {pat:.2f}  roles combined {roles:.2f}  unassigned {un:.2f}  roles+unassigned {roles+un:.2f}")
    print(f"   first speaker = doctor in {fsd.sum()}/{len(d)} recordings")
    for name,g in (("first speaker correct",ok),("first speaker wrong",bad)):
        if len(g): print(f"   {name}: n={len(g)}  SA-WER {100*g.sa_err.sum()/g.sa_n.sum():.2f}  cpWER {100*g.cp_err.sum()/g.cp_n.sum():.2f}  roles+unassigned {100*(g.doc_err.sum()+g.pat_err.sum()+g.unassigned_words.sum())/g.sa_n.sum():.2f}")
    print("   cp_n == sa_n:", (d.cp_n==d.sa_n).all(), "| doc_n+pat_n == sa_n:", ((d.doc_n+d.pat_n)==d.sa_n).all())
