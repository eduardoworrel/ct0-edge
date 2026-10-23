import { headDecide, headAnswer } from "../dist/headdecide.js";
let pass=0,fail=0; const ok=(c,m)=>{if(c){pass++;console.log("  ✓",m);}else{fail++;console.log("  ✗ FAIL:",m);}};

// runtime falso: devolve hidden sintetico constante
const H=2560;
const rt={ hiddens: async (p)=>({atDecide:new Float32Array(H).fill(0.1), atLast:new Float32Array(H).fill(0.1)}) };
const D=8;
const hd={q:new Float32Array(D*H).fill(0.01),qb:new Float32Array(D),k:new Float32Array(D*H).fill(0.01),kb:new Float32Array(D),
          headDim:D,hiddenSize:H,temperature:2,scoreOffset:0};

console.log("headDecide (contrato decide):");
const dec=headDecide(hd,rt);
const p=await dec("state","question");
ok(typeof p==="number" && p>=0 && p<=1, "retorna P em [0,1]");
ok(p>0.5, "score positivo -> P>0.5");

console.log("\nheadAnswer:");
const a1=await headAnswer(hd,rt,{type:"noul",state:"s",question:"q"});
ok(a1.type==="noul" && typeof a1.noul==="number", "noul devolve probabilidade");
const a2=await headAnswer(hd,rt,{type:"score",state:"s",question:"q",levels:5});
ok(a2.score>=1 && a2.score<=5, "score dentro de 1..K");
const a3=await headAnswer(hd,rt,{type:"choice",state:"s",question:"q",options:["a","b","c"]});
ok(Object.keys(a3.choice).length===3, "choice devolve 3 opcoes");
ok(Math.abs(Object.values(a3.choice).reduce((s,v)=>s+v,0)-1)<1e-6, "choice soma 1");

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
