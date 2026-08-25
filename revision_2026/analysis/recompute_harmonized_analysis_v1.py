#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

PE=["learned","sinusoidal","rope","alibi"]
SEEDS=[42,123,456,789,1011,1213]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--package-root",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=200000)
    ap.add_argument("--rng-seed",type=int,default=20260824)
    args=ap.parse_args()
    root=args.package_root
    res=root/"RESULTS"; out=root/"DERIVED"
    e2=pd.read_csv(res/"E2C_VITB_CONTROLLED_2X2"/"DERIVED"/"E2C_2X2_SEED_LEVEL.csv")
    e6s=pd.read_csv(res/"E6_VITS_CONTROLLED_2X2"/"DERIVED"/"E6_VITS_SEED_LEVEL_L4_AND_BASELINES.csv")
    e6l=pd.read_csv(res/"E6_VITS_CONTROLLED_2X2"/"DERIVED"/"E6_VITS_LAYER_LEVEL.csv")
    e6rep=json.loads((res/"E6_VITS_CONTROLLED_2X2"/"DERIVED"/"E6_VITS_REPLICATION_REPORT.json").read_text())
    rng=np.random.default_rng(args.rng_seed); B=args.bootstrap

    def m2(kind,L):
        return np.array([[float(e2[(e2.pe_type==p)&(e2.seed==s)][f"{kind}_L{L}"].iloc[0]) for s in SEEDS] for p in PE])
    def m6(kind,L):
        col={"F2C":"F_to_C_AUROC","C2F":"C_to_F_AUROC","DELTA":"Delta_asym"}[kind]
        return np.array([[float(e6l[(e6l.pe_type==p)&(e6l.seed==s)&(e6l.layer==L)][col].iloc[0]) for s in SEEDS] for p in PE])
    def boot(M):
        idx=rng.integers(0,6,size=(B,4,6))
        v=np.stack([M[i,idx[:,i,:]].mean(1) for i in range(4)],1).mean(1)
        return {"mean":float(M.mean()),"ci95":[float(x) for x in np.quantile(v,[.025,.975])],
                "bootstrap_resamples":B,"rng_seed":args.rng_seed,
                "estimand":"equal-weight fixed four prespecified PE families; seeds resampled within family"}

    harm={"analysis_id":"HARMONIZED_FIXED_PE_L4_v1","status":"POST_AUDIT_ESTIMAND_HARMONIZATION",
          "primary_layer":"L4","ViT_B":{},"ViT_S":{},"architecture_difference_ViTS_minus_ViTB":{},
          "original_prelocked_random_PE_E6":{"pooled_attention_delta":e6rep["ViT_S"]["Attention"]["pooled"],
                                             "replication_gate":e6rep["replication_gate"]}}
    for key,mf in [("ViT_B",m2),("ViT_S",m6)]:
        for metric in ["F2C","C2F","DELTA"]: harm[key][metric]=boot(mf(metric,4))
    idx=rng.integers(0,6,size=(B,4,6))
    for metric in ["F2C","C2F","DELTA"]:
        b,s=m2(metric,4),m6(metric,4)
        v=np.stack([(s[i,idx[:,i,:]]-b[i,idx[:,i,:]]).mean(1) for i in range(4)],1).mean(1)
        harm["architecture_difference_ViTS_minus_ViTB"][metric]={
            "mean":float((s-b).mean()),"ci95":[float(x) for x in np.quantile(v,[.025,.975])],
            "bootstrap_resamples":B,"paired_by":"PE family and seed"}
    (out/"HARMONIZED_FIXED_PE_L4_ANALYSIS.json").write_text(json.dumps(harm,indent=2)+"\n")

    rows=[]
    for arch,mf in [("ViT-B",m2),("ViT-S",m6)]:
        for L in range(1,13):
            F,C=mf("F2C",L),mf("C2F",L)
            for i,p in enumerate(PE):
                rows.append([arch,L,p,float(F[i].mean()),float(C[i].mean()),int((F[i]<.5).sum()),int((C[i]<.5).sum()),int(((F[i]<.5)|(C[i]<.5)).sum()),6])
            rows.append([arch,L,"ALL_4_PE",float(F.mean()),float(C.mean()),int((F<.5).sum()),int((C<.5).sum()),int(((F<.5)|(C<.5)).sum()),24])
    pd.DataFrame(rows,columns=["architecture","layer","pe_type","F2C_mean_AUROC","C2F_mean_AUROC",
        "F2C_inverted_lt_0p5","C2F_inverted_lt_0p5","any_direction_inverted","n_seed_cells"]).to_csv(
        out/"LAYER_SCORE_ORIENTATION_AUDIT.csv",index=False)
    print("POST_AUDIT_RECOMPUTE=PASS")

if __name__=="__main__":
    main()
