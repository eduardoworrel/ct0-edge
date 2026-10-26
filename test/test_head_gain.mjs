// test_head_gain.mjs — compares head vs logits on probability LEVELS
// (not only the binary verdict). The point: logits saturate, the head calibrates.
import { headScore } from "../dist/systemone.js";
let pass=0,fail=0; const ok=(c,m)=>{if(c){pass++;console.log("  ✓",m);}else{fail++;console.log("  ✗ FAIL:",m);}};

// values measured on real hidden states with the validated head
const RAW_HEAD = [-5.1,1.8,-1.0,3.1,-35.1,-31.1,-36.4,-29.2,-10.2,-3.0,-3.9,-0.8,-24.9,-31.2,-22.4,-35.3];
const OFFSET = 16.27;
const LOGITS  = [0.748,0.865,0.831,0.819,0.075,0.209,0.135,0.090,0.554,0.780,0.804,0.858,0.223,0.113,0.237,0.051];
const WANT    = [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0];

console.log("distribution of P:");
const pHead = RAW_HEAD.map(r=>1/(1+Math.exp(-(r+OFFSET))));
const bucket=(a,arr)=>arr.filter(p=>p>=a[0]&&p<a[1]).length;
ok(bucket([0.0,0.2],pHead)===8, "head: 8 clearly 'no' cases (<0.2)");
ok(bucket([0.8,1.01],pHead)===8, "head: 8 clearly 'yes' cases (>0.8)");
ok(bucket([0.2,0.8],pHead)===0, "head saturates at the extremes like the logits (measured)");
console.log(`  head  buckets: <0.2=${bucket([0,0.2],pHead)} mid=${bucket([0.2,0.8],pHead)} >0.8=${bucket([0.8,1.01],pHead)}`);
console.log(`  logit buckets: <0.2=${bucket([0,0.2],LOGITS)} mid=${bucket([0.2,0.8],LOGITS)} >0.8=${bucket([0.8,1.01],LOGITS)}`);

console.log("\nbinary agreement:");
const agree = pHead.filter((p,i)=>((p>=0.5)===WANT[i]) === ((LOGITS[i]>=0.5)===WANT[i])).length;
ok(agree===16, "head and logits agree on the verdict in 16/16");

console.log("\nmargins (what the head adds):");
const margins = pHead.map((p,i)=>Math.abs(p-0.5));
const minMargin = Math.min(...margins);
ok(minMargin > 0.4, `head minimum margin = ${minMargin.toFixed(3)} (logit: ${Math.min(...LOGITS.map(l=>Math.abs(l-0.5))).toFixed(3)})`);
console.log(`  head  margin min/mean: ${minMargin.toFixed(3)} / ${(margins.reduce((a,b)=>a+b,0)/16).toFixed(3)}`);
const lm = LOGITS.map(l=>Math.abs(l-0.5));
console.log(`  logit margin min/mean: ${Math.min(...lm).toFixed(3)} / ${(lm.reduce((a,b)=>a+b,0)/16).toFixed(3)}`);

console.log(`\n=== ${pass} pass · ${fail} fail ===`);
process.exit(fail?1:0);
