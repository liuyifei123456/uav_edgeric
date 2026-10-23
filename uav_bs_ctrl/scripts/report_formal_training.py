import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

root=Path("/workspace/uav_edgeric_stage1/results/stage3/formal_training")
ev=pd.read_csv(root/"cross_evaluation/cross_evaluation.csv")
out=root/"reports"; figdir=root/"figures"; out.mkdir(exist_ok=True); figdir.mkdir(exist_ok=True)
metrics=["TotalThroughput","FairIdx","AvgGlobalUtility","GTServiceFraction","UAVCollisionRate","RBUtilization","MeanGTRate"]
rows=[]
for (trained,evalm),g in ev.groupby(["trained_mode","eval_mode"]):
    row={"trained_mode":trained,"eval_mode":evalm,"n_episodes":len(g)}
    for m in metrics: row[m+"_mean"]=g[m].mean(); row[m+"_std"]=g[m].std(ddof=1)
    rows.append(row)
pd.DataFrame(rows).to_csv(out/"cross_eval_summary.csv",index=False)
per=ev.groupby(["trained_mode","train_seed","eval_mode"])[metrics].mean().reset_index()
per.to_csv(out/"per_training_seed_means.csv",index=False)

def ci(vals):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]
    if len(vals)<2: return float(np.mean(vals)),float("nan"),float("nan")
    mean=vals.mean(); se=stats.sem(vals); half=stats.t.ppf(.975,len(vals)-1)*se
    return mean,mean-half,mean+half

paired=[]
for trained in ["original","edgeric_pf"]:
    a=ev[(ev.trained_mode==trained)&(ev.eval_mode=="original")].set_index(["train_seed","eval_seed"])
    b=ev[(ev.trained_mode==trained)&(ev.eval_mode=="edgeric_pf")].set_index(["train_seed","eval_seed"])
    for m in metrics:
        d=(b[m]-a[m]).dropna()
        mean,lo,hi=ci(d)
        paired.append({"comparison":f"{trained}: PF - Original","metric":m,"n_pairs":len(d),
          "mean_delta":mean,"ci95_low":lo,"ci95_high":hi,
          "relative_percent":100*mean/a[m].mean()})
for evalm in ["original","edgeric_pf"]:
    a=ev[(ev.trained_mode=="original")&(ev.eval_mode==evalm)].set_index(["train_seed","eval_seed"])
    b=ev[(ev.trained_mode=="edgeric_pf")&(ev.eval_mode==evalm)].set_index(["train_seed","eval_seed"])
    for m in metrics:
        d=(b[m]-a[m]).dropna(); mean,lo,hi=ci(d)
        paired.append({"comparison":f"PF-trained - Original-trained @ {evalm}","metric":m,"n_pairs":len(d),
          "mean_delta":mean,"ci95_low":lo,"ci95_high":hi,
          "relative_percent":100*mean/a[m].mean()})
pd.DataFrame(paired).to_csv(out/"paired_differences_ci95.csv",index=False)

# Training curves (each seed separately and mean +/- range)
curve_rows=[]
for mode in ["original","edgeric_pf"]:
    for seed in [0,1,2]:
        p=root/mode/f"seed{seed}"/"progress.txt"
        if p.exists():
            d=pd.read_csv(p,sep="\t")
            d["mode"]=mode; d["seed"]=seed
            curve_rows.append(d)
curves=pd.concat(curve_rows,ignore_index=True)
curves.to_csv(out/"training_curves.csv",index=False)
plot_metrics=[("AverageEpRet","Reward"),("LossQ","Loss"),("TotalThroughput","Throughput"),("FairIdx","Jain"),("AverageAvgGlobalUtility","Utility")]
for col,title in plot_metrics:
    plt.figure(figsize=(7,4))
    for mode,color in [("original","tab:blue"),("edgeric_pf","tab:orange")]:
        q=curves[curves["mode"]==mode].groupby("TotalEnvInteracts")[col].agg(["mean","min","max"]).reset_index()
        plt.plot(q.TotalEnvInteracts,q["mean"],label=mode,color=color)
        plt.fill_between(q.TotalEnvInteracts,q["min"],q["max"],alpha=.15,color=color)
    plt.xlabel("UAV environment steps"); plt.ylabel(title); plt.legend(); plt.tight_layout()
    plt.savefig(figdir/f"training_{col}.png",dpi=160); plt.close()
# 2x2 independent evaluation bars
summary=pd.DataFrame(rows)
for m in ["TotalThroughput","FairIdx","AvgGlobalUtility"]:
    pivot=summary.pivot(index="trained_mode",columns="eval_mode",values=m+"_mean")
    ax=pivot.plot.bar(figsize=(7,4),rot=0); ax.set_ylabel(m); ax.set_xlabel("trained policy")
    ax.figure.tight_layout(); ax.figure.savefig(figdir/f"cross_{m}.png",dpi=160); plt.close(ax.figure)
meta={"training_seeds":[0,1,2],"test_seeds":list(range(2000,2020)),"n_episodes":len(ev),
      "metrics":metrics,"git_sha":None}
(root/"reports"/"statistics_config.json").write_text(json.dumps(meta,indent=2)+"\\n")
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
