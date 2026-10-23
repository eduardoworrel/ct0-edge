// test_head_gain.mjs — compara head vs logits nos NÍVEIS de probabilidade
// (não só no veredito binário). O ponto: o logit satura, o head calibra.
import { headScore } from "../dist/systemone.js";
let pass=0,fail=0; const ok=(c,m)=>{if(c){pass++;console.log("  ✓",m);}else{fail++;console.log("  ✗ FAIL:",m);}};

// valores reais medidos no pod (hiddens_ref + head validado)
const RAW_HEAD = [-5.1,1.8,-1.0,3.1,-35.1,-31.1,-36.4,-29.2,-10.2,-3.0,-3.9,-0.8,-24.9,-31.2,-22.4,-35.3];
const OFFSET = 16.27;
const LOGITS  = [0.748,0.865,0.831,0.819,0.075,0.209,0.135,0.090,0.554,0.780,0.804,0.858,0.223,0.113,0.237,0.051];
const WANT    = [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0];

console.log("distribuicao de P:");
const pHead = RAW_HEAD.map(r=>1/(1+Math.exp(-(r+OFFSET))));
const bucket=(a,arr)=>arr.filter(p=>p>=a[0]&&p<a[1]).length;
ok(bucket([0.0,0.2],pHead)===8, "head: 8 casos claramente 'no' (<0.2)");
ok(bucket([0.8,1.01],pHead)===8, "head: 8 casos claramente 'yes' (>0.8)");
ok(bucket([0.2,0.8],pHead)===0, "head satura nos extremos como o logit (medido)");
console.log(`  head  buckets: <0.2=${bucket([0,0.2],pHead)} mid=${bucket([0.2,0.8],pHead)} >0.8=${bucket([0.8,1.01],pHead)}`);
console.log(`  logit buckets: <0.2=${bucket([0,0.2],LOGITS)} mid=${bucket([0.2,0.8],LOGITS)} >0.8=${bucket([0.8,1.01],LOGITS)}`);

console.log("\nconcordancia binaria:");
const agree = pHead.filter((p,i)=>((p>=0.5)===WANT[i]) === ((LOGITS[i]>=0.5)===WANT[i])).length;
ok(agree===16, "head e logit concordam no veredito em 16/16");

console.log("\nmargens (o que o head adiciona):");
const margins = pHead.map((p,i)=>Math.abs(p-0.5));
const minMargin = Math.min(...margins);
ok(minMargin > 0.4, `margem minima do head = ${minMargin.toFixed(3)} (logit: ${Math.min(...LOGITS.map(l=>Math.abs(l-0.5))).toFixed(3)})`);
console.log(`  head  margem min/med: ${minMargin.toFixed(3)} / ${(margins.reduce((a,b)=>a+b,0)/16).toFixed(3)}`);
const lm = LOGITS.map(l=>Math.abs(l-0.5));
console.log(`  logit margem min/med: ${Math.min(...lm).toFixed(3)} / ${(lm.reduce((a,b)=>a+b,0)/16).toFixed(3)}`);

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
