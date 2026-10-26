// test_systemone.mjs — validates the JS head implementation against reference values
// and the module's own math (no model).
import { headScore, answer, loadHead } from "../dist/systemone.js";
import { readFileSync, existsSync } from "node:fs";

let pass=0, fail=0;
const ok=(c,m)=>{ if(c){pass++;console.log("  ✓",m);} else {fail++;console.log("  ✗ FAIL:",m);} };

console.log("math:");
const D=4,H=8;
const q=new Float32Array(D*H).fill(0.25), qb=new Float32Array(D).fill(0.5);
const k=new Float32Array(D*H).fill(0.25), kb=new Float32Array(D);
const hd={q,qb,k,kb,headDim:D,hiddenSize:H,temperature:2,scoreOffset:0};
const a=new Float32Array(H).fill(1);
// manual: proj_q[d] = 0.5 + sum(0.25*1)=0.5+2=2.5 ; proj_k[d]=2.0 ; dot=4*2.5*2=20 ; /2=10
ok(Math.abs(headScore(hd,a,a)-10)<1e-5, "headScore matches the manual calculation (10)");
ok(Math.abs(answer(hd,{type:"noul"},{atDecide:a,atLast:a}).noul - 1/(1+Math.exp(-10)))<1e-9, "noul = sigmoid(score)");

console.log("\nchoice (softmax over options):");
const r=answer(hd,{type:"choice",options:["x","y","z"]},{atDecide:a,atLast:a,perOption:[a,a,a]});
const vals=Object.values(r.choice);
ok(vals.length===3, "3 options");
ok(Math.abs(vals.reduce((s,v)=>s+v,0)-1)<1e-6, "probabilities sum to 1");
ok(vals.every(v=>Math.abs(v-1/3)<1e-6), "equal -> uniform");

console.log("\nchoice with a missing option:");
const r2=answer(hd,{type:"choice",options:["x","y"]},{atDecide:a,atLast:a,perOption:[a,null]});
ok(r2.choice["x"]===1 && r2.choice["y"]===0, "option without hidden -> prob 0");

console.log("\nchoice: query/key roles (regression for the swap bug):");
// The head is a POINTER HEAD: the QUERY is the <decide> hidden (one per question), each option is a KEY.
// An earlier version passed the option as query and `atLast` as key -> 3/8 instead of 8/8.
// Test: only option "y" has a hidden matching the query -> y must win, and the choice
// must NOT depend on `atLast` (which takes no part in choice).
const D2=4, H2=8;
const qq=new Float32Array(D2*H2).fill(0.25), qqb=new Float32Array(D2).fill(0.5);
const kk=new Float32Array(D2*H2);
for(let d=0;d<D2;d++) kk[d*H2+0]=1;            // k only reads dimension 0
const hd2={q:qq,qb:qqb,k:kk,kb:new Float32Array(D2),headDim:D2,hiddenSize:H2,temperature:2,scoreOffset:0};
const hq=new Float32Array(H2).fill(1);          // query: hidden at <decide>
const hOpt={x:new Float32Array(H2), y:new Float32Array(H2).fill(1), z:new Float32Array(H2)};
const rOpt=answer(hd2,{type:"choice",options:["x","y","z"]},
  {atDecide:hq, atLast:new Float32Array(H2), perOption:[hOpt.x,hOpt.y,hOpt.z]});
// scores: y=5, x=z=0  ->  softmax = e^5/(e^5+2) = 0.9867
ok(rOpt.choice["y"]>0.98, `the option matching the query wins (y=${rOpt.choice["y"].toFixed(3)})`);
// `atLast` must be irrelevant for choice: changing it must change nothing.
const rOpt2=answer(hd2,{type:"choice",options:["x","y","z"]},
  {atDecide:hq, atLast:new Float32Array(H2).fill(99), perOption:[hOpt.x,hOpt.y,hOpt.z]});
ok(rOpt2.choice["y"]===rOpt.choice["y"], "choice does not depend on atLast (k comes from the option)");

console.log("\nordinal score:");
const s1=answer(hd,{type:"score",levels:5},{atDecide:a,atLast:a});
ok(s1.score===5, "high score -> K (5)");
const neg={...hd, q:new Float32Array(D*H).fill(-0.25)};
const s2=answer(neg,{type:"score",levels:5},{atDecide:a,atLast:a});
ok(s2.score===1, "low score -> 1");

console.log("\nloadHead (real head binary):");
const headPath=process.env.CT0_HEAD;
if (!headPath || !existsSync(headPath)) console.log("  SKIP: set CT0_HEAD to a local head.bin to run this section");
else try {
  const buf=readFileSync(headPath);
  const ab=buf.buffer.slice(buf.byteOffset, buf.byteOffset+buf.byteLength);
  const W=loadHead(ab);
  ok(W.headDim===256 && W.hiddenSize===2560, "correct dims (256x2560)");
  ok(Math.abs(W.temperature-2.0)<1e-6, "temperature=2.0");
  ok(W.q.length===256*2560, "q loaded");
  const nz=Array.from({length:256},(_,i)=>Array.from(W.k.subarray(i*2560,(i+1)*2560)).some(v=>v!==0)).filter(Boolean).length;
  ok(nz===228, `k has 228 non-zero rows (recovered), got ${nz}`);
  ok(W.kb.every(v=>v===0), "kb zeroed (lost to truncation)");
  ok(Math.abs(W.scoreOffset-16.27)<0.1, "scoreOffset=+16.27 (recovered via the median, validated 16/16)");
} catch(e){ ok(false, "loadHead: "+e.message); }

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
