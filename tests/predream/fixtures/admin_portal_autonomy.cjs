// Exercise the shipped selection handler and HTTP client, including failure recovery.
const fs = require('node:fs');
const vm = require('node:vm');
async function main() {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const source = fs.readFileSync(input.script, 'utf8');
  const requests = [];
  let control = {autonomy:'manual',auto_routes:['memory'],min_score:0.9,revision:1};
  let failures = input.failures || [];
  const context = vm.createContext({
    URLSearchParams, structuredClone, location:{search:'',hash:''},
    document:{addEventListener(){},createElement(){return {textContent:'',get innerHTML(){return this.textContent;}};}},
    window:{addEventListener(){}},
    async fetch(path, options = {}) {
      const body = options.body ? JSON.parse(options.body) : null;
      requests.push({path,method:options.method || 'GET',body});
      let status = 200;
      let result = {};
      if (path === '/api/v1/control' || path === '/api/v1/control/approve') {
        if (options.method === 'PUT' || options.method === 'POST') {
          status = failures.shift() || 200;
          if (status === 200) control = {...body,revision:control.revision+1};
        }
        result = status === 200 ? control : {error:'Internal permission details must stay out of the page'};
      } else if (path === '/api/v1/system') result = {features:{dreams:{enabled:false}}};
      else if (path === '/api/v1/policy') result = {policy:{}};
      else if (path === '/openapi.json') result = input.spec;
      else result = {items:[]};
      return {status,ok:status===200,text:async()=>JSON.stringify(result),json:async()=>result};
    },
  });
  const boot = '\nnavigate();\nrefresh();\n';
  if (!source.endsWith(boot)) throw new Error('Portal bootstrap changed');
  vm.runInContext(source.slice(0,-boot.length),context);
  context.control = control;
  context.input = input;
  vm.runInContext(`
    state.data={control}; state.token='fictional-test-session'; state.spec=input.spec;
    render=function(){}; toast=function(){};
    openDrawer=function(){throw new Error('Selecting a level opened a confirmation');};
  `,context);
  const states=[];
  for (let index = 0; index < input.levels.length; index += 1) {
    context.level=input.levels[index];
    context.action=(input.approveAt || []).includes(index) ? 'autonomy-approve' : 'autonomy-select';
    await vm.runInContext("handleAction(action,level)",context);
    states.push(vm.runInContext('({control:structuredClone(controlValues()),issue:state.autonomyIssue,html:renderAutonomy(),error:state.error})',context));
  }
  const layers=[];
  for (const advanced of [false,true]) {
    context.advanced=advanced;
    layers.push(vm.runInContext('state.advanced=advanced; ({advanced,html:renderAutonomy()})',context));
  }
  process.stdout.write(JSON.stringify({requests,states,layers}));
}
main().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
