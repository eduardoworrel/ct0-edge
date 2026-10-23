// test_systemone.mjs — valida a implementacao JS do head contra valores de referencia
// e a matematica do proprio modulo (sem modelo).
import { headScore, answer, loadHead } from "../dist/systemone.js";
import { readFileSync } from "node:fs";

let pass=0, fail=0;
const ok=(c,m)=>{ if(c){pass++;console.log("  ✓",m);} else {fail++;console.log("  ✗ FAIL:",m);} };

console.log("matematica:");
const D=4,H=8;
const q=new Float32Array(D*H).fill(0.25), qb=new Float32Array(D).fill(0.5);
const k=new Float32Array(D*H).fill(0.25), kb=new Float32Array(D);
const hd={q,qb,k,kb,headDim:D,hiddenSize:H,temperature:2,scoreOffset:0};
const a=new Float32Array(H).fill(1);
// manual: proj_q[d] = 0.5 + sum(0.25*1)=0.5+2=2.5 ; proj_k[d]=2.0 ; dot=4*2.5*2=20 ; /2=10
ok(Math.abs(headScore(hd,a,a)-10)<1e-5, "headScore bate o calculo manual (10)");
ok(Math.abs(answer(hd,{type:"noul"},{atDecide:a,atLast:a}).noul - 1/(1+Math.exp(-10)))<1e-9, "noul = sigmoid(score)");

console.log("\nchoice (softmax sobre opcoes):");
const r=answer(hd,{type:"choice",options:["x","y","z"]},{atDecide:a,atLast:a,perOption:[a,a,a]});
const vals=Object.values(r.choice);
ok(vals.length===3, "3 opcoes");
ok(Math.abs(vals.reduce((s,v)=>s+v,0)-1)<1e-6, "probabilidades somam 1");
ok(vals.every(v=>Math.abs(v-1/3)<1e-6), "iguais -> uniforme");

console.log("\nchoice com opcao faltando:");
const r2=answer(hd,{type:"choice",options:["x","y"]},{atDecide:a,atLast:a,perOption:[a,null]});
ok(r2.choice["x"]===1 && r2.choice["y"]===0, "opcao sem hidden -> prob 0");

console.log("\nscore ordinal:");
const s1=answer(hd,{type:"score",levels:5},{atDecide:a,atLast:a});
ok(s1.score===5, "score alto -> K (5)");
const neg={...hd, q:new Float32Array(D*H).fill(-0.25)};
const s2=answer(neg,{type:"score",levels:5},{atDecide:a,atLast:a});
ok(s2.score===1, "score baixo -> 1");

console.log("\nloadHead (binario real do pod):");
try {
  const buf=readFileSync("/Users/pc/cai/alias3-mlc/head.bin");
  const ab=buf.buffer.slice(buf.byteOffset, buf.byteOffset+buf.byteLength);
  const W=loadHead(ab);
  ok(W.headDim===256 && W.hiddenSize===2560, "dims corretos (256x2560)");
  ok(Math.abs(W.temperature-2.0)<1e-6, "temperature=2.0");
  ok(W.q.length===256*2560, "q carregado");
  const nz=Array.from({length:256},(_,i)=>Array.from(W.k.subarray(i*2560,(i+1)*2560)).some(v=>v!==0)).filter(Boolean).length;
  ok(nz===228, `k com 228 linhas nao-zero (recuperadas), obteve ${nz}`);
  ok(W.kb.every(v=>v===0), "kb zerado (perdido no truncamento)");
  ok(Math.abs(W.scoreOffset-16.27)<0.1, "scoreOffset=+16.27 (recuperado pela mediana, validado 16/16)");
} catch(e){ ok(false, "loadHead: "+e.message); }

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
