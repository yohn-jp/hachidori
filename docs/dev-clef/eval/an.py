import json,collections
R=json.load(open("vocab_raw.json"))
Q={}
exec(open("vocab.py").read().split("# name")[1].split("def post")[0].split("\n",1)[1].replace("V={","V={",1),Q) if False else None
txt={}
import re
src=open("vocab.py").read()
for m in re.finditer(r'\("([shi]\d)","([^"]+)",(\d)\)',src): txt[m.group(1)]=(m.group(2),int(m.group(3)))
def pred(r): 
    yes=r["choice"]=="yes"
    return (not yes) if r["inv"] else yes
S=collections.defaultdict(lambda:collections.defaultdict(list))
for r in R: S[r["id"]][r["split"]].append(r)
def stats(rs):
    n=len(rs);ok=sum(pred(r)==bool(r["gold"]) for r in rs)
    fn=sum(bool(r["gold"]) and not pred(r) for r in rs);fp=sum((not r["gold"]) and pred(r) for r in rs)
    cc=[r["conf"] for r in rs if pred(r)==bool(r["gold"])]
    return ok,n,fn,fp,(sum(cc)/len(cc) if cc else 0)
names={"s":"範囲外","i":"不可逆","h":"共有"}
for k in "sih":
    ids=sorted(i for i in S if i[0]==k)
    print(f"\n=== {names[k]} ===  dev正解 / test正解 / 合計  見逃し/誤警報(合計)  語数  確信度(正解時)")
    rows=[]
    for i in ids:
        d=stats(S[i]["dev"]);t=stats(S[i]["test"]);a=stats(S[i]["dev"]+S[i]["test"])
        rows.append((a[0],d[0],i,d,t,a))
    for _,_,i,d,t,a in sorted(rows,key=lambda x:(-x[5][0],-x[5][4])):
        w=len(txt[i][0].split())
        print(f"{i} {d[0]:2d}/{d[1]} {t[0]:2d}/{t[1]}  {a[0]:2d}/{a[1]}  FN{a[2]:2d} FP{a[3]:2d}  {w:2d}語  {a[4]:.2f}  {'(反転)' if txt[i][1] else ''}{txt[i][0]}")
