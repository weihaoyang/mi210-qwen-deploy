import urllib.request as u,json,time,hashlib,sys,concurrent.futures
def go(k):
 p=('This is a benchmark of GPU inference on a Linux server. ' * (4 if k<4 else 180))+'Write a long numbered list of practical Linux administration tips with explanations. '+str(k)
 d=dict(prompt=p,n_predict=128,temperature=0,cache_prompt=False,ignore_eos=True)
 t=time.monotonic()
 r=json.load(u.urlopen(u.Request('http://127.0.0.1:'+sys.argv[1]+'/completion',json.dumps(d).encode(),{'Content-Type':'application/json'}),timeout=65))
 print(k,round(time.monotonic()-t,2),r['timings'],hashlib.sha256(r['content'].encode()).hexdigest()[:12],flush=True)
if len(sys.argv) != 2:
    raise SystemExit('Usage: python3 scripts/benchmark.py PORT')
go(0)
with concurrent.futures.ThreadPoolExecutor(4) as e: list(e.map(go,[1,2,3,4]))
