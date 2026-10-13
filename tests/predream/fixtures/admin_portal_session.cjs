// Run the actual hosted-session discovery, settings and connection UI.
const fs = require('node:fs');
const vm = require('node:vm');
async function main() {
  const input = JSON.parse(fs.readFileSync(0,'utf8'));
  const source = fs.readFileSync(input.script,'utf8');
  const requests = [], drawers = [];
  const context = vm.createContext({
    URLSearchParams, URL, structuredClone, location:{search:'',hash:'',origin:'http://localhost'},
    document:{addEventListener(){},createElement(){return {textContent:'',get innerHTML(){return this.textContent;}};}},
    window:{addEventListener(){}},
    async fetch(path, options={}) {
      const body = options.body ? JSON.parse(options.body) : null;
      requests.push({path,method:options.method || 'GET',headers:options.headers,body});
      let value = {items:[]};
      if (path.startsWith('/session/')) {
        if (body?.session_seconds) input.status.session_seconds=body.session_seconds;
        value=input.status;
      } else if (path.endsWith('/control')) value={autonomy:'manual',revision:1};
      else if (path.endsWith('/system')) value={features:{dreams:{enabled:false}}};
      else if (path.endsWith('/policy')) value={policy:{}};
      else if (path.endsWith('openapi.json')) value={paths:{}};
      return {status:200,ok:true,text:async()=>JSON.stringify(value),json:async()=>value};
    },
  });
  const boot='\nnavigate();\nrefresh();\n';
  if (!source.endsWith(boot)) throw new Error('Portal bootstrap changed');
  vm.runInContext(source.slice(0,-boot.length),context);
  context.input=input; context.drawers=drawers;
  vm.runInContext(`render=function(){};toast=function(){};openDrawer=function(title,html){drawers.push({title,html});};`,context);
  for (let i=0;i<3;i+=1) await vm.runInContext('refresh()',context);
  const dataAvailable=vm.runInContext('state.data!==null',context);
  const welcome=vm.runInContext('renderWelcome()',context);
  const layers=[];
  for (const advanced of [false,true]) {
    context.advanced=advanced;
    const html=vm.runInContext('state.advanced=advanced;portalAccessSettings()',context);
    await vm.runInContext('connectionPanel(true)',context);
    layers.push({advanced,html,connection:drawers.at(-1).html});
  }
  let error=null;
  if (input.timeout!==undefined) {
    context.timeout=input.timeout;
    try {await vm.runInContext('saveSessionTimeout({elements:{seconds:{value:timeout},minutes:{value:input.minutes}}})',context);}
    catch (failure) {error=failure.message;}
  }
  if (input.logoutOthers) await vm.runInContext("handleAction('portal-logout-others','')",context);
  process.stdout.write(JSON.stringify({requests,dataAvailable,welcome,layers,error}));
}
main().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
