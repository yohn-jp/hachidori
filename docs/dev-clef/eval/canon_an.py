"""Summarize canon_raw.json and compare with the decomposed canon questions in vocab_raw.json."""
import json,collections,statistics as st
R=[r for r in json.load(open("canon_raw.json")) if "err" not in r]
V=list(dict.fromkeys(r["v"] for r in R))
print("variant  dev  test  total  FN FP  mean_ms  mean_chars  conf_when_correct")
for v in V:
    rs=[r for r in R if r["v"]==v]
    ok=lambda x:sum(r["pred"]==r["gold"] for r in x)
    d=[r for r in rs if r["split"]=="dev"];t=[r for r in rs if r["split"]=="test"]
    fn=sum(r["gold"]==1 and r["pred"]==0 for r in rs);fp=sum(r["gold"]==0 and r["pred"]==1 for r in rs)
    cc=[r["conf"] for r in rs if r["pred"]==r["gold"]]
    print(f"{v} {ok(d)}/30 {ok(t)}/30 {ok(rs)}/60 {fn} {fp} {st.mean(r['ms'] for r in rs):.0f} {st.mean(r['chars'] for r in rs):.0f} {st.mean(cc):.2f}")
print("\nper combination (scope/irreversible/shared):")
for cb in sorted({tuple(r["combo"]) for r in R}):
    row=[]
    for v in V:
        rs=[r for r in R if r["v"]==v and tuple(r["combo"])==cb];row.append(f"{sum(r['pred']==r['gold'] for r in rs)}/{len(rs)}")
    print(cb," ".join(row))
# decomposed comparator: OR of approval.scope (s5), approval.irreversible (i3), approval.shared (h3)
V2=json.load(open("vocab_raw.json"))
by=collections.defaultdict(dict)
for r in V2:
    if r["id"] in ("s5","i3","h3"):
        yes=r["choice"]=="yes"; by[(r["split"],r["case"])][r["id"]]=((not yes) if r["inv"] else yes, r["gold"])
for sp in ("dev","test"):
    ok=n=0
    for (s,c),d in by.items():
        if s==sp:
            n+=1;ok+=int(any(g for _,g in d.values()))==int(any(x for x,_ in d.values()))
    print(sp,"decomposed OR:",f"{ok}/{n}")
