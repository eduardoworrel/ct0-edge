import { strongEvidence } from "/Users/pc/cai/ct0-lib/dist/recon.js";
import { BrowserTool } from "/Users/pc/cai/agent-browser-lite/dist/browser.js";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
// por que TODA página dispara tudo? olha a saída do inspect numa página limpa
const html=readFileSync("/Users/pc/cai/alias3-mlc/scenarios/clean-csp.html","utf8");
const bt=new BrowserTool(new JSDOM(html,{url:"https://t.example/"}).window.document,{});
for(const t of ["inspect","storage","cookies","resources","sinks","forms","messages","events","csp","clobber","links","iframes"]){
  const o=String(await bt.toolset()[t]({}));
  const e=strongEvidence(o);
  console.log(`${t.padEnd(11)} ev="${e}"  ${o.slice(0,110).replace(/\n/g," ")}`);
}
