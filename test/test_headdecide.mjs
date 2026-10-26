import { headDecide, headAnswer } from "../dist/headdecide.js";
let pass=0,fail=0; const ok=(c,m)=>{if(c){pass++;console.log("  ✓",m);}else{fail++;console.log("  ✗ FAIL:",m);}};

// fake runtime: returns a constant synthetic hidden
const H=2560;
const hv=()=>new Float32Array(H).fill(0.1);
const rt={ hiddens: async (p)=>({atDecide:hv(), atLast:hv()}),
          optHiddens: async (p)=>({atDecide:hv(), atLast:hv(), atOpts:[hv(),hv(),hv()]}) };
const D=8;
const hd={q:new Float32Array(D*H).fill(0.01),qb:new Float32Array(D),k:new Float32Array(D*H).fill(0.01),kb:new Float32Array(D),
          headDim:D,hiddenSize:H,temperature:2,scoreOffset:0};

console.log("headDecide (decide contract):");
const dec=headDecide(hd,rt);
const p=await dec("state","question");
ok(typeof p==="number" && p>=0 && p<=1, "returns P in [0,1]");
ok(p>0.5, "positive score -> P>0.5");

console.log("\nheadAnswer:");
const a1=await headAnswer(hd,rt,{type:"noul",state:"s",question:"q"});
ok(a1.type==="noul" && typeof a1.noul==="number", "noul returns a probability");
const a2=await headAnswer(hd,rt,{type:"score",state:"s",question:"q",levels:5});
ok(a2.score>=1 && a2.score<=5, "score within 1..K");
const a3=await headAnswer(hd,rt,{type:"choice",state:"s",question:"q",options:["a","b","c"]});
ok(Object.keys(a3.choice).length===3, "choice returns 3 options");
ok(Math.abs(Object.values(a3.choice).reduce((s,v)=>s+v,0)-1)<1e-6, "choice sums to 1");

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
