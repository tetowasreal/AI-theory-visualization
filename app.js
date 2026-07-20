/* =========================================================
   딥러닝 실험실 — shared utilities
   ========================================================= */

// ---- seeded RNG (mulberry32) for reproducible "random" demos ----
function makeRng(seed){
  let a = seed >>> 0;
  return function(){
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng){
  // Box-Muller
  let u = 0, v = 0;
  while(u === 0) u = rng();
  while(v === 0) v = rng();
  return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
}
const sigmoid = x => 1/(1+Math.exp(-x));
const dsigmoid = x => { const s = sigmoid(x); return s*(1-s); };
const relu = x => Math.max(0,x);
const drelu = x => x > 0 ? 1 : 0;
const tanh = x => Math.tanh(x);
const dtanh = x => 1 - Math.tanh(x)*Math.tanh(x);
const clamp = (v,a,b) => Math.max(a, Math.min(b,v));
const lerp = (a,b,t) => a+(b-a)*t;
const fmt = (v,d=3) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d);

// ---- color ramp for heatmaps ----
function rampColor(t){ // t in [0,1] -> [r,g,b]
  const stops = [
    [8,22,34],   // deep navy (low)
    [24,64,90],  // steel blue
    [61,140,150],// teal
    [255,180,84],// amber (high)
  ];
  t = clamp(t,0,1);
  const n = stops.length-1;
  const seg = Math.min(n-1, Math.floor(t*n));
  const localT = t*n - seg;
  const c0 = stops[seg], c1 = stops[seg+1];
  return [
    Math.round(lerp(c0[0],c1[0],localT)),
    Math.round(lerp(c0[1],c1[1],localT)),
    Math.round(lerp(c0[2],c1[2],localT)),
  ];
}

// ---- simple linear-system solver (Gaussian elimination) for polyfit ----
function solveLinearSystem(A, b){
  const n = A.length;
  const M = A.map((row,i)=> row.concat([b[i]]));
  for(let col=0; col<n; col++){
    let piv = col;
    for(let r=col+1;r<n;r++) if(Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col],M[piv]] = [M[piv],M[col]];
    if(Math.abs(M[col][col]) < 1e-12) continue;
    for(let r=0;r<n;r++){
      if(r===col) continue;
      const f = M[r][col]/M[col][col];
      for(let c=col;c<=n;c++) M[r][c] -= f*M[col][c];
    }
  }
  return M.map((row,i)=> Math.abs(row[i])<1e-12 ? 0 : row[n]/row[i]);
}
function polyFit(xs, ys, degree){
  const n = degree+1;
  const A = Array.from({length:n},()=>Array(n).fill(0));
  const b = Array(n).fill(0);
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      let s=0; for(let k=0;k<xs.length;k++) s += Math.pow(xs[k], i+j);
      A[i][j]=s;
    }
    let s=0; for(let k=0;k<xs.length;k++) s += ys[k]*Math.pow(xs[k], i);
    b[i]=s;
  }
  return solveLinearSystem(A,b); // coefficients [c0, c1, ... cn] for c0 + c1 x + ...
}
function polyEval(coeffs, x){
  let s=0, p=1;
  for(let i=0;i<coeffs.length;i++){ s+= coeffs[i]*p; p*=x; }
  return s;
}

/* =========================================================
   Graph — canvas coordinate helper for 2D plots
   ========================================================= */
class Graph{
  constructor(canvas, {xmin,xmax,ymin,ymax,pad=36,heightCss=280}){
    this.canvas = canvas;
    this.xmin=xmin; this.xmax=xmax; this.ymin=ymin; this.ymax=ymax; this.pad=pad;
    this.heightCss = heightCss;
    this.resize();
    this.ctx = canvas.getContext('2d');
  }
  resize(){
    const cssW = this.canvas.parentElement.clientWidth;
    const cssH = this.heightCss;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.style.width = cssW+'px';
    this.canvas.style.height = cssH+'px';
    this.canvas.width = cssW*dpr;
    this.canvas.height = cssH*dpr;
    this.w = cssW; this.h = cssH;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  X(x){ const p=this.pad; return p + (x-this.xmin)/(this.xmax-this.xmin) * (this.w-2*p); }
  Y(y){ const p=this.pad; return this.h-p - (y-this.ymin)/(this.ymax-this.ymin) * (this.h-2*p); }
  clear(){
    const ctx=this.ctx;
    ctx.fillStyle = '#081019';
    ctx.fillRect(0,0,this.w,this.h);
  }
  axes({xlabel='x', ylabel='y'}={}){
    const ctx=this.ctx, p=this.pad;
    ctx.strokeStyle = '#1c3441'; ctx.lineWidth=1;
    ctx.font = '10px IBM Plex Mono, monospace';
    // grid
    const xTicks = 6, yTicks = 5;
    ctx.fillStyle = '#5d7683';
    for(let i=0;i<=xTicks;i++){
      const x = this.xmin + (this.xmax-this.xmin)*i/xTicks;
      const px = this.X(x);
      ctx.beginPath(); ctx.moveTo(px,p); ctx.lineTo(px,this.h-p); ctx.stroke();
      ctx.fillText(x.toFixed(1), px-10, this.h-p+14);
    }
    for(let i=0;i<=yTicks;i++){
      const y = this.ymin + (this.ymax-this.ymin)*i/yTicks;
      const py = this.Y(y);
      ctx.beginPath(); ctx.moveTo(p,py); ctx.lineTo(this.w-p,py); ctx.stroke();
      ctx.fillText(y.toFixed(1), 4, py+3);
    }
    // zero axes emphasized
    ctx.strokeStyle = '#2c4a58'; ctx.lineWidth=1.4;
    if(this.xmin<=0 && this.xmax>=0){
      ctx.beginPath(); ctx.moveTo(this.X(0),p); ctx.lineTo(this.X(0),this.h-p); ctx.stroke();
    }
    if(this.ymin<=0 && this.ymax>=0){
      ctx.beginPath(); ctx.moveTo(p,this.Y(0)); ctx.lineTo(this.w-p,this.Y(0)); ctx.stroke();
    }
  }
  plot(fn, {color='#ffb454', width=2, dash=[]}={}){
    const ctx = this.ctx, p=this.pad;
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash);
    ctx.beginPath();
    let started=false;
    for(let px=p; px<=this.w-p; px++){
      const x = this.xmin + (px-p)/(this.w-2*p)*(this.xmax-this.xmin);
      const y = fn(x);
      if(!isFinite(y)){ started=false; continue; }
      const py = this.Y(clamp(y,this.ymin-1e6,this.ymax+1e6));
      if(!started){ ctx.moveTo(px,py); started=true; } else ctx.lineTo(px,py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  point(x,y,{color='#63d9c4', r=4}={}){
    const ctx=this.ctx;
    ctx.fillStyle=color;
    ctx.beginPath(); ctx.arc(this.X(x),this.Y(y),r,0,7); ctx.fill();
  }
  line(x1,y1,x2,y2,{color='#e7f0f2', width=1.4, dash=[]}={}){
    const ctx=this.ctx;
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(this.X(x1),this.Y(y1)); ctx.lineTo(this.X(x2),this.Y(y2)); ctx.stroke();
    ctx.setLineDash([]);
  }
  vband(x1,x2,color='rgba(255,180,84,.12)'){
    const ctx=this.ctx, p=this.pad;
    ctx.fillStyle=color;
    ctx.fillRect(this.X(x1),p,this.X(x2)-this.X(x1),this.h-2*p);
  }
  hband(y1,y2,color='rgba(99,217,196,.12)'){
    const ctx=this.ctx, p=this.pad;
    ctx.fillStyle=color;
    ctx.fillRect(p,this.Y(y2),this.w-2*p,this.Y(y1)-this.Y(y2));
  }
}

/* =========================================================
   small DOM helpers for building controls
   ========================================================= */
function el(tag, attrs={}, children=[]){
  const e = document.createElement(tag);
  for(const k in attrs){
    if(k === 'class') e.className = attrs[k];
    else if(k === 'html') e.innerHTML = attrs[k];
    else if(k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  children.forEach(c => {
    if(c === null || c === undefined) return;
    if(c instanceof Node) e.appendChild(c);
    else e.appendChild(document.createTextNode(String(c)));
  });
  return e;
}

function makeSlider({label, min, max, step, value, onInput, fmt2=(v)=>v}){
  const wrap = el('div',{class:'control'});
  const lab = el('label',{}, [label+' ', el('span',{class:'val'}, [fmt2(value)])]);
  const input = el('input',{type:'range', min, max, step, value});
  input.addEventListener('input', ()=>{
    const v = parseFloat(input.value);
    lab.querySelector('.val').textContent = fmt2(v);
    onInput(v);
  });
  wrap.appendChild(lab); wrap.appendChild(input);
  return {wrap, input};
}
function makeSelect({label, options, value, onChange}){
  const wrap = el('div',{class:'control'});
  const lab = el('label',{}, [label]);
  const sel = el('select',{});
  options.forEach(o=>{
    const opt = el('option',{value:o.value}, [o.label]);
    if(o.value===value) opt.setAttribute('selected','selected');
    sel.appendChild(opt);
  });
  sel.addEventListener('change', ()=> onChange(sel.value));
  wrap.appendChild(lab); wrap.appendChild(sel);
  return {wrap, sel};
}
function panel(titleText){
  const p = el('div',{class:'panel'});
  p.appendChild(el('div',{class:'panel-title'},[titleText]));
  return p;
}

/* =========================================================
   CHAPTER 1 — 딥러닝 개요
   ========================================================= */
const topic1_1 = {
  id:'1-1', title:'1-1강. 모두를 위한 딥러닝 개요',
  desc:`인공지능(AI), 머신러닝(ML), 딥러닝(DL)은 <b>포함 관계</b>에 있습니다.
  AI는 "인간처럼 사고/행동하는 기계"라는 가장 큰 목표이고, 그 목표를 데이터로부터
  <b>스스로 규칙을 학습</b>해서 달성하는 방법론이 머신러닝, 그리고 그 중에서도
  다층 <b>인공신경망</b>을 사용하는 방법이 딥러닝입니다. 원을 클릭해서 각 개념의 설명을 확인해보세요.`,
  render(root){
    const p = panel('AI ⊃ ML ⊃ DL');
    const wrap = el('div',{class:'canvas-wrap'});
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS,'svg');
    svg.setAttribute('viewBox','0 0 600 320');
    svg.style.width='100%'; svg.style.display='block'; svg.style.background='#081019';
    const rings = [
      {r:150, cx:300, cy:170, color:'#24404f', label:'AI (인공지능)', info:'인간의 지능적 행동을 기계로 구현하려는 가장 큰 목표. 규칙 기반 시스템도 AI에 포함됩니다.'},
      {r:105, cx:300, cy:185, color:'#3d6b78', label:'ML (머신러닝)', info:'규칙을 직접 짜지 않고, 데이터로부터 패턴을 스스로 학습하는 AI의 하위 분야.'},
      {r:60, cx:300, cy:200, color:'#ffb454', label:'DL (딥러닝)', info:'여러 층(layer)의 인공신경망을 쌓아 표현력을 극대화한 머신러닝 기법.'},
    ];
    rings.forEach(r=>{
      const c = document.createElementNS(svgNS,'circle');
      c.setAttribute('cx',r.cx); c.setAttribute('cy',r.cy); c.setAttribute('r',r.r);
      c.setAttribute('fill', r.color); c.setAttribute('fill-opacity','0.35');
      c.setAttribute('stroke', r.color); c.setAttribute('stroke-width','1.5');
      c.style.cursor='pointer';
      c.addEventListener('click', ()=>{ infoBox.textContent = r.info; infoBox.style.color = r.color; });
      svg.appendChild(c);
    });
    rings.forEach(r=>{
      const t = document.createElementNS(svgNS,'text');
      t.setAttribute('x', r.cx); t.setAttribute('y', r.cy - r.r - 10);
      t.setAttribute('fill', '#e7f0f2'); t.setAttribute('font-family','IBM Plex Mono, monospace');
      t.setAttribute('font-size','13'); t.setAttribute('text-anchor','middle');
      t.textContent = r.label;
      svg.appendChild(t);
    });
    wrap.appendChild(svg);
    wrap.style.padding='14px 0';
    p.appendChild(wrap);
    const infoBox = el('div',{class:'readout'},['원을 클릭하면 설명이 여기에 표시됩니다.']);
    p.appendChild(infoBox);
    root.appendChild(p);
  }
};

const topic1_2 = {
  id:'1-2', title:"1-2강. 경사하강법 vs Newton's Method",
  desc:`둘 다 함수의 최솟값을 찾아가는 <b>최적화 알고리즘</b>입니다.
  경사하강법은 <code>x ← x - η·f'(x)</code> 로 기울기 방향으로 조금씩 내려가고,
  Newton's Method는 <code>x ← x - f'(x)/f''(x)</code> 로 곡률(2차 미분)까지 활용해 더 빠르게 수렴합니다.
  학습률과 시작점을 바꿔가며 두 경로를 비교해보세요.`,
  render(root){
    const p = panel('f(x) = 0.3x⁴ − x² + 0.1x  위에서의 하강 경로');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas');
    cw.appendChild(canvas); p.appendChild(cw);

    const f  = x => 0.3*x**4 - x**2 + 0.1*x;
    const df = x => 1.2*x**3 - 2*x + 0.1;
    const d2f= x => 3.6*x**2 - 2;

    let lr = 0.08, x0 = 1.6, steps = 12;
    let gdPath=[], nPath=[];

    function compute(){
      gdPath=[x0]; let xg=x0;
      for(let i=0;i<steps;i++){ xg = xg - lr*df(xg); gdPath.push(xg); }
      nPath=[x0]; let xn=x0;
      for(let i=0;i<steps;i++){
        const d2 = d2f(xn); const step = Math.abs(d2)<0.3 ? 0.3*Math.sign(df(xn)||1) : df(xn)/d2;
        xn = xn - step; nPath.push(xn);
      }
    }
    let graph;
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(f, {color:'#3d6b78', width:2});
      for(let i=0;i<gdPath.length-1;i++){
        graph.line(gdPath[i], f(gdPath[i]), gdPath[i+1], f(gdPath[i+1]), {color:'#ffb454', width:1.6});
      }
      gdPath.forEach((x,i)=> graph.point(x,f(x),{color:'#ffb454', r: i===gdPath.length-1?5:2.5}));
      for(let i=0;i<nPath.length-1;i++){
        graph.line(nPath[i], f(nPath[i]), nPath[i+1], f(nPath[i+1]), {color:'#63d9c4', width:1.6, dash:[4,3]});
      }
      nPath.forEach((x,i)=> graph.point(x,f(x),{color:'#63d9c4', r: i===nPath.length-1?5:2.5}));
    }
    function rebuild(){ compute(); draw(); }

    graph = new Graph(canvas, {xmin:-2.2,xmax:2.2,ymin:-1.6,ymax:3.5,heightCss:300});
    window.addEventListener('resize', ()=>{ graph.resize(); draw(); });
    rebuild();

    p.appendChild(el('div',{class:'legend'},[
      el('span',{},[el('i',{style:'background:#ffb454'}), '경사하강법 (Gradient Descent)']),
      el('span',{},[el('i',{style:'background:#63d9c4'}), "Newton's Method"]),
    ]));

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'학습률 η (경사하강법)', min:0.01,max:0.3,step:0.01,value:lr,
      fmt2:v=>v.toFixed(2), onInput:v=>{ lr=v; rebuild(); }}).wrap);
    controls.appendChild(makeSlider({label:'시작점 x₀', min:-2,max:2,step:0.05,value:x0,
      fmt2:v=>v.toFixed(2), onInput:v=>{ x0=v; rebuild(); }}).wrap);
    controls.appendChild(makeSlider({label:'반복 횟수', min:1,max:25,step:1,value:steps,
      fmt2:v=>v, onInput:v=>{ steps=v; rebuild(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'});
    p.appendChild(rd);
    const updReadout = ()=>{
      rd.innerHTML = `경사하강법 도착: <b>x=${fmt(gdPath.at(-1))}</b> · Newton 도착: <b>x=${fmt(nPath.at(-1))}</b> · 전역 최소점 근처: x≈±1.3, x=0`;
    };
    const origRebuild = rebuild;
    rebuild = function(){ origRebuild(); updReadout(); };
    updReadout();
    p.appendChild(el('div',{class:'note'},['💡 Newton\'s Method가 곡률 정보를 이용해 더 적은 스텝으로 수렴하지만, 이차미분이 0에 가까우면(변곡점 부근) 불안정해질 수 있어요.']));
    root.appendChild(p);
  }
};

const topic1_3 = {
  id:'1-3', title:'1-3강. 자기지도학습 (Self-Supervised Learning)',
  desc:`정답 라벨 없이, <b>데이터 그 자체에서 정답을 만들어내는</b> 학습 방식입니다.
  대표적으로 이미지의 일부를 가리고(masking) 나머지 부분으로부터 가려진 부분을 예측하게 만드는
  "pretext task"가 있습니다. 아래 그리드에서 마스킹 비율을 조절하고 "가려보기 → 복원해보기"를 눌러보세요.`,
  render(root){
    const p = panel('마스킹 기반 pretext task 시뮬레이션 (8×8 패턴)');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    const N=8;
    const rng = makeRng(7);
    const truth = [];
    for(let i=0;i<N;i++){ truth.push([]); for(let j=0;j<N;j++){
      const cx=3.5, cy=3.5;
      const d = Math.hypot(i-cx,j-cy);
      const v = 0.5 + 0.5*Math.sin(d*1.1) ;
      truth[i].push(clamp(v,0,1));
    }}
    let maskRatio = 0.35;
    let mask = null;
    let revealed = false;

    function cellColor(v){ const [r,g,b]=rampColor(v); return `rgb(${r},${g},${b})`; }

    function draw(){
      const wCss = canvas.parentElement.clientWidth;
      const size = Math.min(wCss, 420);
      const dpr = window.devicePixelRatio||1;
      canvas.style.width=size+'px'; canvas.style.height=size+'px';
      canvas.width=size*dpr; canvas.height=size*dpr;
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,size,size);
      const cell = size/N;
      for(let i=0;i<N;i++) for(let j=0;j<N;j++){
        const hidden = mask && mask[i][j];
        let v = truth[i][j];
        if(hidden && !revealed) v = null;
        if(hidden && revealed){
          let s=0,c=0;
          [[i-1,j],[i+1,j],[i,j-1],[i,j+1]].forEach(([a,b])=>{
            if(a>=0&&a<N&&b>=0&&b<N){ s+=truth[a][b]; c++; }
          });
          v = c? s/c : 0.5;
        }
        ctx.fillStyle = v===null ? '#16212c' : cellColor(v);
        ctx.fillRect(j*cell+1, i*cell+1, cell-2, cell-2);
        if(hidden){
          ctx.strokeStyle = revealed ? '#63d9c4' : '#ffb454';
          ctx.lineWidth=1.5;
          ctx.strokeRect(j*cell+1, i*cell+1, cell-2, cell-2);
        }
      }
    }
    function newMask(){
      mask = Array.from({length:N},()=>Array.from({length:N},()=> rng()<maskRatio));
      revealed = false; draw();
    }
    newMask();
    window.addEventListener('resize', draw);

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'마스킹 비율', min:0.1,max:0.7,step:0.05,value:maskRatio,
      fmt2:v=>Math.round(v*100)+'%', onInput:v=>{ maskRatio=v; newMask(); }}).wrap);
    p.appendChild(controls);
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:newMask},['🎲 새로 가리기']));
    btnRow.appendChild(el('button',{class:'btn secondary', onclick:()=>{ revealed=true; draw(); }},['🧠 모델이 복원해보기']));
    p.appendChild(btnRow);
    p.appendChild(el('div',{class:'note'},['💡 실제 자기지도학습(BERT의 masked-token 예측, MAE의 masked-patch 예측 등)은 신경망이 대량의 데이터로부터 이 복원 규칙 자체를 학습합니다. 여기서는 이웃 평균이라는 단순 규칙으로 "가려진 것을 맥락으로 추측한다"는 핵심 아이디어만 보여줍니다.']));
    root.appendChild(p);
  }
};

const topic1_4 = {
  id:'1-4', title:'1-4강. 강화학습 (Reinforcement Learning)',
  desc:`에이전트가 <b>환경과 상호작용</b>하며 보상(reward)을 최대화하는 행동을 스스로 찾아가는 학습 방식입니다.
  아래는 5×5 그리드월드에서 <b>가치반복(Value Iteration)</b>으로 각 칸의 가치를 계산하고,
  그 가치가 높은 방향으로 이동하는 정책을 실행하는 예시입니다. 할인율 γ를 바꿔보세요.`,
  render(root){
    const p = panel('그리드월드 · 가치반복(Value Iteration) 기반 정책');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    const SIZE=5;
    const goal=[4,4], obstacle=[[2,2],[1,3]];
    let gamma=0.9;
    let V, agent=[0,0], reward=0, running=false, timer=null;

    function isObstacle(i,j){ return obstacle.some(([a,b])=>a===i&&b===j); }
    function valueIteration(){
      V = Array.from({length:SIZE},()=>Array(SIZE).fill(0));
      for(let it=0; it<80; it++){
        const Vn = V.map(r=>r.slice());
        for(let i=0;i<SIZE;i++) for(let j=0;j<SIZE;j++){
          if(i===goal[0]&&j===goal[1]){ Vn[i][j]=10; continue; }
          if(isObstacle(i,j)){ Vn[i][j]=-5; continue; }
          const neighbors=[[i-1,j],[i+1,j],[i,j-1],[i,j+1]].filter(([a,b])=>a>=0&&a<SIZE&&b>=0&&b<SIZE);
          let best=-Infinity;
          neighbors.forEach(([a,b])=>{ best = Math.max(best, -1 + gamma*V[a][b]); });
          Vn[i][j]= neighbors.length? best : 0;
        }
        V=Vn;
      }
    }
    function bestMove(i,j){
      const neighbors=[[i-1,j,'↑'],[i+1,j,'↓'],[i,j-1,'←'],[i,j+1,'→']].filter(([a,b])=>a>=0&&a<SIZE&&b>=0&&b<SIZE);
      let best=null, bv=-Infinity;
      neighbors.forEach(([a,b,d])=>{ if(V[a][b]>bv){ bv=V[a][b]; best=[a,b,d]; } });
      return best;
    }
    function draw(){
      const wCss = canvas.parentElement.clientWidth;
      const size = Math.min(wCss, 420);
      const dpr = window.devicePixelRatio||1;
      canvas.style.width=size+'px'; canvas.style.height=size+'px';
      canvas.width=size*dpr; canvas.height=size*dpr;
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,size,size);
      const cell=size/SIZE;
      let minV=Infinity,maxV=-Infinity;
      V.forEach(r=>r.forEach(v=>{minV=Math.min(minV,v);maxV=Math.max(maxV,v);}));
      for(let i=0;i<SIZE;i++) for(let j=0;j<SIZE;j++){
        const t = (V[i][j]-minV)/((maxV-minV)||1);
        const [r,g,b]=rampColor(t);
        ctx.fillStyle=isObstacle(i,j) ? '#3a1c22' : `rgb(${r},${g},${b})`;
        ctx.fillRect(j*cell+1,i*cell+1,cell-2,cell-2);
        ctx.fillStyle='#0b1620'; ctx.font='9px IBM Plex Mono, monospace';
        if(!isObstacle(i,j)) ctx.fillText(V[i][j].toFixed(1), j*cell+4, i*cell+13);
      }
      ctx.fillStyle='#ffb454'; ctx.font='bold 16px sans-serif';
      ctx.fillText('★', goal[1]*cell+cell/2-8, goal[0]*cell+cell/2+6);
      ctx.beginPath(); ctx.fillStyle='#63d9c4';
      ctx.arc(agent[1]*cell+cell/2, agent[0]*cell+cell/2, cell*0.22, 0, 7); ctx.fill();
    }
    function reset(){ agent=[0,0]; reward=0; updReadout(); draw(); }
    function step(){
      if(agent[0]===goal[0] && agent[1]===goal[1]){ stop(); return; }
      const mv = bestMove(agent[0],agent[1]);
      if(mv){ agent=[mv[0],mv[1]]; reward += (agent[0]===goal[0]&&agent[1]===goal[1]) ? 10 : -1; }
      updReadout(); draw();
    }
    function stop(){ running=false; clearInterval(timer); timer=null; }
    function updReadout(){
      rd.innerHTML = `누적 보상: <b>${reward}</b> · γ(할인율): <b>${gamma.toFixed(2)}</b> · 에이전트 위치: <b>(${agent[0]},${agent[1]})</b>`;
    }

    valueIteration(); 
    window.addEventListener('resize', draw);

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'할인율 γ', min:0.5,max:0.99,step:0.01,value:gamma,
      fmt2:v=>v.toFixed(2), onInput:v=>{ gamma=v; valueIteration(); draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); 
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>{ if(!running){ running=true; timer=setInterval(step,450); } }},['▶ 정책 실행']));
    btnRow.appendChild(el('button',{class:'btn secondary', onclick:reset},['↺ 초기화']));
    p.appendChild(btnRow);
    p.appendChild(rd);
    reset();
    p.appendChild(el('div',{class:'note'},['💡 색이 밝을수록(진한 amber) 그 칸의 가치 V(s)가 높습니다 — 에이전트는 항상 이웃 중 가치가 가장 높은 칸으로 이동합니다. 빨간 칸은 장애물(패널티), 별표는 목표(+10 보상)입니다.']));
    root.appendChild(p);
  }
};

const CHAPTER_1 = { id:'1', label:'1강 · 딥러닝이란', topics:[topic1_1, topic1_2, topic1_3, topic1_4] };

/* =========================================================
   CHAPTER 2 — 인공신경망 기초
   ========================================================= */
const topic2_1 = {
  id:'2-1', title:'2-1강 · 2-2강. 인공신경망 (Artificial Neural Network)',
  desc:`인공신경망의 기본 단위는 <b>뉴런(perceptron)</b>입니다. 입력 <code>x</code>에 가중치
  <code>w</code>를 곱하고 편향 <code>b</code>를 더한 뒤(<code>z = w·x+b</code>), <b>활성화함수</b>를 통과시켜
  출력 <code>a = σ(z)</code>를 만듭니다. 이런 뉴런을 여러 층으로 쌓은 것이 인공신경망입니다.
  슬라이더로 값을 바꾸면 아래 네트워크 다이어그램의 계산이 실시간으로 갱신됩니다.`,
  render(root){
    const p = panel('단일 뉴런 계산 + 2층 신경망 순전파(forward pass)');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    let x1=0.6, x2=-0.3, w1=1.2, w2=0.8, b=0.1, actName='sigmoid';
    const acts = { sigmoid, relu, tanh, linear: x=>x };

    // simple fixed 2-input -> 2-hidden -> 1-output net, driven by x1,x2
    const W1 = [[0.8,-0.5],[0.4,0.9]], B1=[0.1,-0.2];
    const W2 = [0.7,-0.6], B2=0.15;

    function draw(){
      const wCss = canvas.parentElement.clientWidth;
      const h = 300;
      const dpr = window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);

      const act = acts[actName];
      // hidden layer
      const z1 = [
        W1[0][0]*x1 + W1[0][1]*x2 + B1[0],
        W1[1][0]*x1 + W1[1][1]*x2 + B1[1],
      ];
      const a1 = z1.map(act);
      const z2 = W2[0]*a1[0] + W2[1]*a1[1] + B2;
      const a2 = act(z2);

      const inX=70, hidX=wCss/2, outX=wCss-90;
      const inY=[h*0.32,h*0.68], hidY=[h*0.28,h*0.72], outY=h*0.5;

      function node(x,y,label,val,color){
        ctx.beginPath(); ctx.arc(x,y,22,0,7);
        ctx.fillStyle='#101f2b'; ctx.fill();
        ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
        ctx.fillStyle='#e7f0f2'; ctx.font='10px IBM Plex Mono, monospace'; ctx.textAlign='center';
        ctx.fillText(label, x, y-30);
        ctx.fillStyle=color; ctx.font='11px IBM Plex Mono, monospace';
        ctx.fillText(val, x, y+4);
        ctx.textAlign='left';
      }
      function edge(x1_,y1_,x2_,y2_,w){
        ctx.strokeStyle = w>=0 ? 'rgba(255,180,84,'+clamp(Math.abs(w),0.15,0.9)+')' : 'rgba(255,107,107,'+clamp(Math.abs(w),0.15,0.9)+')';
        ctx.lineWidth = clamp(Math.abs(w)*3,1,5);
        ctx.beginPath(); ctx.moveTo(x1_,y1_); ctx.lineTo(x2_,y2_); ctx.stroke();
      }
      // edges input->hidden
      edge(inX,inY[0],hidX,hidY[0],W1[0][0]); edge(inX,inY[0],hidX,hidY[1],W1[1][0]);
      edge(inX,inY[1],hidX,hidY[0],W1[0][1]); edge(inX,inY[1],hidX,hidY[1],W1[1][1]);
      edge(hidX,hidY[0],outX,outY,W2[0]); edge(hidX,hidY[1],outX,outY,W2[1]);

      node(inX,inY[0],'x1', x1.toFixed(2), '#63d9c4');
      node(inX,inY[1],'x2', x2.toFixed(2), '#63d9c4');
      node(hidX,hidY[0],'h1', a1[0].toFixed(2), '#ffb454');
      node(hidX,hidY[1],'h2', a1[1].toFixed(2), '#ffb454');
      node(outX,outY,'ŷ', a2.toFixed(2), '#b48bf2');

      window._nn_out = {z1,a1,z2,a2};
    }
    draw(); window.addEventListener('resize', draw);

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'입력 x1', min:-2,max:2,step:0.05,value:x1, fmt2:v=>v.toFixed(2),
      onInput:v=>{ x1=v; draw(); updRd(); }}).wrap);
    controls.appendChild(makeSlider({label:'입력 x2', min:-2,max:2,step:0.05,value:x2, fmt2:v=>v.toFixed(2),
      onInput:v=>{ x2=v; draw(); updRd(); }}).wrap);
    controls.appendChild(makeSelect({label:'활성화함수', value:actName,
      options:[{value:'sigmoid',label:'Sigmoid'},{value:'relu',label:'ReLU'},{value:'tanh',label:'Tanh'},{value:'linear',label:'Linear (활성화 없음)'}],
      onChange:v=>{ actName=v; draw(); updRd(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    function updRd(){
      const o = window._nn_out;
      rd.innerHTML = `z1=[${o.z1.map(v=>v.toFixed(2)).join(', ')}] · a1=[${o.a1.map(v=>v.toFixed(2)).join(', ')}] · z2=${o.z2.toFixed(2)} · ŷ=${o.a2.toFixed(2)}`;
    }
    updRd();
    p.appendChild(el('div',{class:'note'},['💡 은닉층 가중치(W1,W2)는 이 데모에서는 고정되어 있고, 입력과 활성화함수만 바꿔가며 "순전파(forward pass)"가 층을 따라 어떻게 값을 변환해나가는지 확인하는 데 집중했어요.']));
    root.appendChild(p);
  }
};

const topic2_3 = {
  id:'2-3', title:'2-3강. 선형회귀 (Linear Regression)',
  desc:`데이터 점들을 가장 잘 설명하는 직선 <code>ŷ = wx + b</code>를 찾는 문제입니다.
  손실함수로는 보통 <b>평균제곱오차(MSE)</b> <code>= (1/n)Σ(y-ŷ)²</code>를 사용합니다.
  w, b를 직접 조절해서 손실을 최소화해보고, "최적해 찾기"로 최소제곱법 정답과 비교해보세요.`,
  render(root){
    const p = panel('직선 피팅과 잔차(residual)');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    const rng = makeRng(42);
    const xs=[], ys=[];
    for(let i=0;i<14;i++){ const x=lerp(-2,2,i/13); xs.push(x); ys.push(2*x+1+gaussian(rng)*0.6); }
    let w=1, b=0;
    let graph;
    function mse(){ let s=0; for(let i=0;i<xs.length;i++){ const e=ys[i]-(w*xs[i]+b); s+=e*e; } return s/xs.length; }
    function draw(){
      graph.clear(); graph.axes();
      xs.forEach((x,i)=>{
        graph.line(x, ys[i], x, w*x+b, {color:'rgba(255,107,107,.5)', width:1});
        graph.point(x, ys[i], {color:'#63d9c4', r:3.5});
      });
      graph.plot(x=> w*x+b, {color:'#ffb454', width:2.2});
      rd.innerHTML = `w=<b>${w.toFixed(2)}</b> · b=<b>${b.toFixed(2)}</b> · MSE=<b>${mse().toFixed(3)}</b>`;
    }
    graph = new Graph(canvas, {xmin:-2.5,xmax:2.5,ymin:-4,ymax:6,heightCss:300});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});

    const controls = el('div',{class:'controls'});
    const wSlider = makeSlider({label:'기울기 w', min:-3,max:3,step:0.05,value:w, fmt2:v=>v.toFixed(2),
      onInput:v=>{ w=v; draw(); }});
    const bSlider = makeSlider({label:'절편 b', min:-3,max:3,step:0.05,value:b, fmt2:v=>v.toFixed(2),
      onInput:v=>{ b=v; draw(); }});
    controls.appendChild(wSlider.wrap); controls.appendChild(bSlider.wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>{
      const c = polyFit(xs,ys,1); b=c[0]; w=c[1];
      wSlider.input.value=w; bSlider.input.value=b;
      wSlider.wrap.querySelector('.val').textContent=w.toFixed(2);
      bSlider.wrap.querySelector('.val').textContent=b.toFixed(2);
      draw();
    }},['⚡ 최적해 찾기 (최소제곱법)']));
    p.appendChild(btnRow);
    draw();
    root.appendChild(p);
  }
};

const topic2_4 = {
  id:'2-4', title:'2-4강. 경사하강법 (2차원 손실 곡면)',
  desc:`손실함수가 여러 개의 파라미터를 가질 때, 경사하강법은 <b>손실 곡면 위에서 기울기(그래디언트)의
  반대 방향</b>으로 이동하며 최솟값을 찾습니다. 여기서는 f(w,b) = w² + 3b² 라는 그릇 모양 손실 곡면에서
  경사하강 경로를 확인해보세요.`,
  render(root){
    const p = panel('f(w,b) = w² + 3b²  등고선 위의 경사하강 경로');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    const f = (w,b)=> w*w + 3*b*b;
    const grad = (w,b)=> [2*w, 6*b];
    let lr=0.15, w0=2.2, b0=1.6, iters=15;
    let graph, path=[];

    function compute(){
      path=[[w0,b0]]; let w=w0,b=b0;
      for(let i=0;i<iters;i++){ const [gw,gb]=grad(w,b); w-=lr*gw; b-=lr*gb; path.push([w,b]); if(Math.abs(w)>1e3) break; }
    }
    function draw(){
      graph.clear(); graph.axes();
      // contour rings
      for(let k=1;k<=8;k++){
        const level = k*k*0.4;
        graph.ctx.strokeStyle = 'rgba(61,140,150,'+(0.15+k*0.03)+')';
        graph.ctx.lineWidth=1;
        graph.ctx.beginPath();
        for(let a=0;a<=64;a++){
          const th = a/64*2*Math.PI;
          const w = Math.sqrt(level)*Math.cos(th);
          const b = Math.sqrt(level/3)*Math.sin(th);
          const px=graph.X(w), py=graph.Y(b);
          if(a===0) graph.ctx.moveTo(px,py); else graph.ctx.lineTo(px,py);
        }
        graph.ctx.stroke();
      }
      for(let i=0;i<path.length-1;i++){
        graph.line(path[i][0],path[i][1],path[i+1][0],path[i+1][1],{color:'#ffb454', width:1.8});
      }
      path.forEach((pt,i)=> graph.point(pt[0],pt[1],{color: i===path.length-1?'#ff6b6b':'#ffb454', r:i===path.length-1?5.5:3}));
      rd.innerHTML = `시작 손실 f=<b>${fmt(f(w0,b0),2)}</b> → 최종 손실 f=<b>${fmt(f(...path.at(-1)),4)}</b> (최소값 0 은 원점)`;
    }
    graph = new Graph(canvas,{xmin:-3,xmax:3,ymin:-3,ymax:3,heightCss:320});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'학습률 η', min:0.01,max:0.35,step:0.01,value:lr, fmt2:v=>v.toFixed(2),
      onInput:v=>{ lr=v; compute(); draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'시작 w₀', min:-2.8,max:2.8,step:0.1,value:w0, fmt2:v=>v.toFixed(1),
      onInput:v=>{ w0=v; compute(); draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'시작 b₀', min:-2.8,max:2.8,step:0.1,value:b0, fmt2:v=>v.toFixed(1),
      onInput:v=>{ b0=v; compute(); draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'반복 횟수', min:1,max:40,step:1,value:iters, fmt2:v=>v,
      onInput:v=>{ iters=v; compute(); draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    p.appendChild(el('div',{class:'note'},['💡 학습률을 너무 크게 올리면(예: 0.3 이상) b축 방향으로 진동하며 발산할 수 있어요 — 곡률이 큰 방향(b, 계수 3)이 더 예민하기 때문입니다.']));
    compute(); draw();
    root.appendChild(p);
  }
};

const topic2_5 = {
  id:'2-5', title:'2-5강. 웨이트 초기화 기법들',
  desc:`가중치를 어떻게 초기화하느냐에 따라 층을 거듭할수록 <b>활성화 값이 0으로 소멸하거나 폭발</b>할 수 있습니다.
  Xavier 초기화(<code>~1/√n</code>)는 sigmoid/tanh에, He 초기화(<code>~√(2/n)</code>)는 ReLU에 적합합니다.
  아래에서 초기화 방식을 바꿔가며 층이 깊어질수록 활성화 분포가 어떻게 변하는지 관찰해보세요.`,
  render(root){
    const p = panel('층별 활성화 값 분포 (6개 층, 뉴런 64개, 고정 시드)');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);

    let scheme='xavier', actName='tanh';
    const LAYERS=6, NEURONS=64;
    const act = { tanh, sigmoid, relu }; 

    function simulate(){
      const rng = makeRng(11);
      let a = Array.from({length:NEURONS},()=> gaussian(rng)); // input activations ~N(0,1)
      const snapshots=[a.slice()];
      for(let l=0;l<LAYERS;l++){
        const n = NEURONS;
        let scale;
        if(scheme==='zero') scale = 0;
        else if(scheme==='large') scale = 1.0;
        else if(scheme==='xavier') scale = Math.sqrt(1/n);
        else scale = Math.sqrt(2/n); // he
        const next = [];
        for(let i=0;i<n;i++){
          let z=0;
          for(let j=0;j<n;j++) z += (scheme==='zero'?0:gaussian(rng)*scale) * a[j];
          next.push(act[actName](z));
        }
        a = next; snapshots.push(a.slice());
      }
      return snapshots;
    }
    function draw(){
      const snaps = simulate();
      const wCss = canvas.parentElement.clientWidth;
      const h=300, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);
      const colW = wCss/snaps.length;
      const bins=20;
      let maxCount=1;
      const histograms = snaps.map(layer=>{
        const hcount = Array(bins).fill(0);
        layer.forEach(v=>{ const idx=clamp(Math.floor((v+1.2)/2.4*bins),0,bins-1); hcount[idx]++; });
        maxCount = Math.max(maxCount, ...hcount);
        return hcount;
      });
      histograms.forEach((hcount,li)=>{
        const x0 = li*colW;
        ctx.strokeStyle='#1c3441'; ctx.strokeRect(x0+4,10,colW-8,h-40);
        hcount.forEach((c,bi)=>{
          const bh = (c/maxCount)*(h-50);
          const by = h-30-bh;
          const bx = x0+6+bi*((colW-12)/bins);
          ctx.fillStyle = li===0 ? '#3d6b78' : '#ffb454';
          ctx.fillRect(bx, by, (colW-12)/bins-1, bh);
        });
        ctx.fillStyle='#93aab5'; ctx.font='10px IBM Plex Mono, monospace'; ctx.textAlign='center';
        ctx.fillText(li===0?'입력':`층 ${li}`, x0+colW/2, h-14);
        ctx.textAlign='left';
      });
      // stats
      const last = snaps.at(-1);
      const mean = last.reduce((a,b)=>a+b,0)/last.length;
      const varr = last.reduce((a,b)=>a+(b-mean)**2,0)/last.length;
      rd.innerHTML = `마지막 층 활성화 — 평균: <b>${mean.toFixed(3)}</b> · 분산: <b>${varr.toFixed(4)}</b> ${varr<1e-4? ' <span style="color:#ff6b6b">(거의 소멸!)</span>':''}`;
    }
    window.addEventListener('resize', draw);

    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSelect({label:'초기화 방식', value:scheme,
      options:[{value:'zero',label:'0으로 초기화'},{value:'large',label:'큰 난수 (N(0,1))'},{value:'xavier',label:'Xavier (1/√n)'},{value:'he',label:'He (√(2/n))'}],
      onChange:v=>{ scheme=v; draw(); }}).wrap);
    controls.appendChild(makeSelect({label:'활성화함수', value:actName,
      options:[{value:'tanh',label:'Tanh'},{value:'sigmoid',label:'Sigmoid'},{value:'relu',label:'ReLU'}],
      onChange:v=>{ actName=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 "0으로 초기화"를 선택하면 모든 뉴런이 완전히 동일하게 학습되는 대칭성 문제가 생기고, "큰 난수"는 층이 깊어질수록 분산이 폭발하거나(ReLU) 포화되어(sigmoid/tanh) 그래디언트가 죽습니다. Xavier/He는 층을 거쳐도 분산을 비슷하게 유지하도록 설계되었어요.']));
    root.appendChild(p);
  }
};

const CHAPTER_2 = { id:'2', label:'2강 · 인공신경망 기초', topics:[topic2_1, topic2_3, topic2_4, topic2_5] };

/* =========================================================
   CHAPTER 3 — 미적분 기초
   ========================================================= */
const topic3_1 = {
  id:'3-1', title:'3-1강. 극한과 입실론-델타 논법',
  desc:`"x가 a에 가까워질 때 f(x)가 L에 가까워진다"는 직관을, <b>ε(오차 허용범위)</b>와
  <b>δ(x의 허용범위)</b>로 엄밀하게 정의한 것이 입실론-델타 논법입니다: "임의의 ε에 대해,
  |x-a|&lt;δ 이면 항상 |f(x)-L|&lt;ε 이 되는 δ를 찾을 수 있다." 슬라이더로 ε을 줄여보며
  그에 맞는 δ 밴드가 어떻게 좁아지는지 확인해보세요.`,
  render(root){
    const p = panel('f(x) = x² ,  a = 1, L = 1 에서의 ε-δ 밴드');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const f = x=>x*x;
    const a=1, L=1;
    let eps=0.4;
    let graph;
    function findDelta(eps){
      // for f(x)=x^2 near a=1, find delta such that |x-1|<delta => |x^2-1|<eps (numeric search)
      let lo=0, hi=1;
      for(let it=0; it<40; it++){
        const mid=(lo+hi)/2;
        const worst = Math.max(Math.abs(f(a+mid)-L), Math.abs(f(a-mid)-L));
        if(worst < eps) lo=mid; else hi=mid;
      }
      return lo;
    }
    function draw(){
      graph.clear(); graph.axes();
      const delta = findDelta(eps);
      graph.hband(L-eps,L+eps);
      graph.vband(a-delta,a+delta);
      graph.plot(f, {color:'#63d9c4', width:2.2});
      graph.line(a-delta,-2,a-delta,4,{color:'#ffb454', width:1, dash:[3,3]});
      graph.line(a+delta,-2,a+delta,4,{color:'#ffb454', width:1, dash:[3,3]});
      graph.line(-2,L-eps,4,L-eps,{color:'#b48bf2', width:1, dash:[3,3]});
      graph.line(-2,L+eps,4,L+eps,{color:'#b48bf2', width:1, dash:[3,3]});
      graph.point(a,L,{color:'#ff6b6b', r:5});
      rd.innerHTML = `ε = <b>${eps.toFixed(2)}</b> 일 때, 필요한 δ = <b>${delta.toFixed(4)}</b> (즉 |x-1|&lt;${delta.toFixed(4)} 이면 |f(x)-1|&lt;${eps.toFixed(2)} 보장)`;
    }
    graph = new Graph(canvas,{xmin:-0.5,xmax:2.5,ymin:-0.5,ymax:4,heightCss:300});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'ε (오차 허용범위)', min:0.05,max:1.5,step:0.05,value:eps,
      fmt2:v=>v.toFixed(2), onInput:v=>{ eps=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 보라색 밴드는 y값의 허용 오차(ε), 주황색 밴드는 그에 대응하는 x의 허용 범위(δ)입니다. ε을 줄일수록 필요한 δ도 함께 줄어드는 게 "극한"의 엄밀한 의미예요.']));
    root.appendChild(p);
  }
};

const topic3_2 = {
  id:'3-2', title:'3-2강. 미분과 도함수',
  desc:`도함수 <code>f'(x)</code>는 "x를 아주 조금(h) 움직였을 때 f(x)가 얼마나 변하는가"의 극한값입니다:
  <code>f'(x) = lim_{h→0} [f(x+h)-f(x)] / h</code>. h가 클 때의 <b>할선(secant line)</b>이
  h를 0으로 줄이면 <b>접선(tangent line)</b>으로 수렴하는 과정을 직접 확인해보세요.`,
  render(root){
    const p = panel('f(x) = sin(x) + 0.15x² 위의 할선 → 접선 수렴');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const f = x => Math.sin(x) + 0.15*x*x;
    const df = x => Math.cos(x) + 0.3*x;
    let x0=0.6, h=1.2;
    let graph;
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(f, {color:'#3d6b78', width:2});
      const slopeSecant = (f(x0+h)-f(x0))/h;
      graph.plot(x=> f(x0) + slopeSecant*(x-x0), {color:'#ffb454', width:1.8, dash:[5,3]});
      graph.point(x0,f(x0),{color:'#63d9c4', r:5});
      graph.point(x0+h,f(x0+h),{color:'#ff6b6b', r:5});
      const trueSlope = df(x0);
      graph.plot(x=> f(x0) + trueSlope*(x-x0), {color:'#b48bf2', width:1.3});
      rd.innerHTML = `할선 기울기 (h=${h.toFixed(2)}): <b>${slopeSecant.toFixed(3)}</b> · 실제 접선 기울기 f'(x₀): <b>${trueSlope.toFixed(3)}</b> · 오차: <b>${Math.abs(slopeSecant-trueSlope).toFixed(4)}</b>`;
    }
    graph = new Graph(canvas,{xmin:-3,xmax:4,ymin:-2.5,ymax:4,heightCss:300});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'x₀ (기준점)', min:-2.5,max:3,step:0.05,value:x0, fmt2:v=>v.toFixed(2),
      onInput:v=>{ x0=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'h (간격, 0에 가까울수록 접선)', min:0.02,max:2.5,step:0.02,value:h, fmt2:v=>v.toFixed(2),
      onInput:v=>{ h=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'legend'},[
      el('span',{},[el('i',{style:'background:#ffb454'}),'할선 (h만큼 떨어진 두 점을 잇는 직선)']),
      el('span',{},[el('i',{style:'background:#b48bf2'}),'실제 접선 (h→0의 극한)']),
    ]));
    root.appendChild(p);
  }
};

const topic3_3 = {
  id:'3-3', title:'3-3강. 연쇄 법칙 (Chain Rule)',
  desc:`합성함수 <code>z = g(y), y = f(x)</code>의 미분은 <code>dz/dx = dz/dy · dy/dx</code> 로 계산됩니다.
  이는 딥러닝의 <b>역전파(backpropagation)</b>를 가능하게 하는 가장 핵심적인 수학 도구입니다.
  아래에서 x를 바꿔가며 각 단계의 값과 미분이 어떻게 곱해지는지 확인해보세요.`,
  render(root){
    const p = panel('y = f(x) = x², z = g(y) = sin(y)  →  dz/dx = cos(y)·2x');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let x=1.0;
    function draw(){
      const y = x*x;
      const z = Math.sin(y);
      const dydx = 2*x;
      const dzdy = Math.cos(y);
      const dzdx = dzdy*dydx;

      const wCss = canvas.parentElement.clientWidth, h=200, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);

      const stages = [
        {label:'x', val:x, sub:''},
        {label:'y = x²', val:y, sub:`dy/dx = 2x = ${dydx.toFixed(2)}`},
        {label:'z = sin(y)', val:z, sub:`dz/dy = cos(y) = ${dzdy.toFixed(2)}`},
      ];
      const gap = wCss/3;
      stages.forEach((s,i)=>{
        const cx = gap*i + gap/2;
        ctx.beginPath(); ctx.arc(cx,h*0.42,32,0,7);
        ctx.fillStyle='#101f2b'; ctx.fill();
        ctx.strokeStyle = i===0?'#63d9c4': i===1?'#ffb454':'#b48bf2'; ctx.lineWidth=2; ctx.stroke();
        ctx.textAlign='center';
        ctx.fillStyle='#e7f0f2'; ctx.font='12px IBM Plex Mono, monospace';
        ctx.fillText(s.label, cx, h*0.42-42);
        ctx.fillStyle= i===0?'#63d9c4': i===1?'#ffb454':'#b48bf2'; ctx.font='13px IBM Plex Mono, monospace';
        ctx.fillText(s.val.toFixed(3), cx, h*0.42+5);
        ctx.fillStyle='#93aab5'; ctx.font='10px IBM Plex Mono, monospace';
        if(s.sub) ctx.fillText(s.sub, cx, h*0.42+55);
        ctx.textAlign='left';
        if(i<2){
          ctx.strokeStyle='#5d7683'; ctx.lineWidth=1.4;
          ctx.beginPath(); ctx.moveTo(cx+34,h*0.42); ctx.lineTo(cx+gap-34,h*0.42); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(cx+gap-34,h*0.42); ctx.lineTo(cx+gap-42,h*0.42-5); ctx.lineTo(cx+gap-42,h*0.42+5); ctx.fill();
        }
      });
      rd.innerHTML = `dz/dx = dz/dy · dy/dx = ${dzdy.toFixed(3)} × ${dydx.toFixed(3)} = <b>${dzdx.toFixed(3)}</b>`;
    }
    window.addEventListener('resize', draw);
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'x', min:-2.5,max:2.5,step:0.05,value:x, fmt2:v=>v.toFixed(2),
      onInput:v=>{ x=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    root.appendChild(p);
  }
};

const topic3_4 = {
  id:'3-4', title:'3-4강. 편미분과 그래디언트',
  desc:`변수가 2개 이상인 함수에서, <b>한 변수만 남기고 나머지를 상수로 고정</b>한 채 미분하는 것이
  편미분입니다. 모든 변수에 대한 편미분을 모은 벡터 <code>∇f = (∂f/∂x, ∂f/∂y)</code>가 <b>그래디언트</b>이며,
  이는 함수가 <b>가장 가파르게 증가하는 방향</b>을 가리킵니다. 점을 옮겨가며 화살표를 확인해보세요.`,
  render(root){
    const p = panel('f(x,y) = x² + xy + y²  의 등고선과 그래디언트 벡터');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const f = (x,y)=> x*x + x*y + y*y;
    const gradf = (x,y)=> [2*x+y, x+2*y];
    let px_=1.2, py_=0.8;
    let graph;
    function draw(){
      graph.clear(); graph.axes();
      // heatmap-ish contour via level curves (numeric sampling of value at grid, draw iso-lines simply via many rings not exact but illustrative)
      for(let lvl=1; lvl<=10; lvl++){
        const level = lvl*0.7;
        graph.ctx.strokeStyle = 'rgba(61,140,150,'+(0.12+lvl*0.02)+')';
        graph.ctx.lineWidth=1;
        graph.ctx.beginPath();
        let started=false;
        for(let a=0;a<=128;a++){
          const th = a/128*2*Math.PI;
          // approximate ellipse for x^2+xy+y^2=level via rotated axes (rough numeric sampling)
          let bestR=null;
          for(let r=0; r<3.5; r+=0.02){
            const x = r*Math.cos(th), y = r*Math.sin(th);
            if(f(x,y) >= level){ bestR=r; break; }
          }
          if(bestR===null) continue;
          const x = bestR*Math.cos(th), y = bestR*Math.sin(th);
          const px=graph.X(x), py=graph.Y(y);
          if(!started){ graph.ctx.moveTo(px,py); started=true; } else graph.ctx.lineTo(px,py);
        }
        graph.ctx.closePath(); graph.ctx.stroke();
      }
      const [gx,gy] = gradf(px_,py_);
      const norm = Math.hypot(gx,gy) || 1;
      const scale = 0.9;
      graph.line(px_,py_, px_+gx/norm*scale, py_+gy/norm*scale, {color:'#ffb454', width:2.4});
      // arrowhead
      const ex = px_+gx/norm*scale, ey = py_+gy/norm*scale;
      graph.point(ex,ey,{color:'#ffb454', r:4});
      graph.point(px_,py_,{color:'#63d9c4', r:5.5});
      rd.innerHTML = `f(${px_.toFixed(2)}, ${py_.toFixed(2)}) = <b>${f(px_,py_).toFixed(3)}</b> · ∇f = (<b>${gx.toFixed(2)}</b>, <b>${gy.toFixed(2)}</b>) — 이 방향이 함수가 가장 빠르게 증가하는 방향`;
    }
    graph = new Graph(canvas,{xmin:-3,xmax:3,ymin:-3,ymax:3,heightCss:320});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'x', min:-2.8,max:2.8,step:0.05,value:px_, fmt2:v=>v.toFixed(2),
      onInput:v=>{ px_=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'y', min:-2.8,max:2.8,step:0.05,value:py_, fmt2:v=>v.toFixed(2),
      onInput:v=>{ py_=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    root.appendChild(p);
  }
};

const CHAPTER_3 = { id:'3', label:'3강 · 미적분 기초', topics:[topic3_1, topic3_2, topic3_3, topic3_4] };

/* =========================================================
   CHAPTER 4 — 확률과 통계 기초
   ========================================================= */
const topic4_1 = {
  id:'4-1', title:'4-1강. 랜덤 변수와 확률 분포',
  desc:`랜덤 변수(random variable)는 <b>무작위 실험의 결과를 숫자로 매핑</b>한 것이고,
  확률 분포는 그 숫자들이 나올 확률을 알려줍니다. 아래에서 주사위(이산)와 동전(베르누이)을
  반복해서 던져보면, 시행 횟수가 늘어날수록 막대그래프가 이론적 분포에 점점 가까워지는 걸
  직접 확인할 수 있습니다 — <b>큰 수의 법칙</b>이에요.`,
  render(root){
    const p = panel('시뮬레이션으로 보는 확률 분포 수렴');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let dist='dice', counts=[0,0,0,0,0,0], nTrials=0;
    const rng = makeRng(3);
    function reset(){
      counts = dist==='dice' ? [0,0,0,0,0,0] : [0,0];
      nTrials=0; draw();
    }
    function sample(n){
      for(let i=0;i<n;i++){
        if(dist==='dice'){ counts[Math.floor(rng()*6)]++; }
        else { counts[rng()<0.5?0:1]++; }
        nTrials++;
      }
      draw();
    }
    function draw(){
      const wCss = canvas.parentElement.clientWidth, h=280, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);
      const n = counts.length;
      const barW = wCss/(n*1.6);
      const theoretical = 1/n;
      const maxFreq = Math.max(0.05, ...counts.map(c=> nTrials? c/nTrials : 0));
      const plotH = h-60;
      for(let i=0;i<n;i++){
        const cx = wCss*(i+0.5)/n;
        const freq = nTrials? counts[i]/nTrials : 0;
        const barH = (freq/Math.max(maxFreq,theoretical*1.6))*plotH;
        ctx.fillStyle = '#ffb454';
        ctx.fillRect(cx-barW/2, h-40-barH, barW, barH);
        // theoretical line marker
        const theoH = (theoretical/Math.max(maxFreq,theoretical*1.6))*plotH;
        ctx.strokeStyle='#63d9c4'; ctx.lineWidth=1.6; ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.moveTo(cx-barW/1.4, h-40-theoH); ctx.lineTo(cx+barW/1.4, h-40-theoH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle='#e7f0f2'; ctx.font='11px IBM Plex Mono, monospace'; ctx.textAlign='center';
        ctx.fillText(dist==='dice'? (i+1) : (i===0?'앞면':'뒷면'), cx, h-18);
        ctx.fillStyle='#93aab5'; ctx.font='10px IBM Plex Mono, monospace';
        ctx.fillText((freq*100).toFixed(1)+'%', cx, h-46-barH);
        ctx.textAlign='left';
      }
      rd.innerHTML = `시행 횟수: <b>${nTrials}</b> · 이론적 확률: 각 <b>${(theoretical*100).toFixed(1)}%</b> (청록 점선) · 시행이 늘수록 주황 막대가 점선에 수렴`;
    }
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSelect({label:'랜덤 변수', value:dist,
      options:[{value:'dice',label:'주사위 (6면, 이산균등분포)'},{value:'coin',label:'동전 (베르누이분포)'}],
      onChange:v=>{ dist=v; reset(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>sample(1)},['🎲 1회 시행']));
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>sample(50)},['🎲 50회 시행']));
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>sample(1000)},['🎲 1000회 시행']));
    btnRow.appendChild(el('button',{class:'btn secondary', onclick:reset},['↺ 초기화']));
    p.appendChild(btnRow);
    window.addEventListener('resize', draw);
    reset();
    root.appendChild(p);
  }
};

const topic4_2 = {
  id:'4-2', title:'4-2강. 평균과 분산',
  desc:`평균(mean)은 데이터의 <b>중심 위치</b>, 분산(variance)은 <b>평균으로부터 얼마나 퍼져 있는가</b>를
  나타냅니다: <code>Var(X) = E[(X-μ)²]</code>. 데이터의 흩어진 정도(퍼짐)를 슬라이더로 조절하면서
  평균은 그대로인데 분산만 달라지는 모습을 관찰해보세요.`,
  render(root){
    const p = panel('숫자 직선 위의 데이터 분포 — 평균 vs 분산');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let spread=1.0;
    const rng = makeRng(5);
    const base = Array.from({length:40},()=> gaussian(rng));
    function draw(){
      const mu=3;
      const data = base.map(z=> mu + z*spread);
      const mean = data.reduce((a,b)=>a+b,0)/data.length;
      const varr = data.reduce((a,b)=>a+(b-mean)**2,0)/data.length;
      const sd = Math.sqrt(varr);

      const wCss = canvas.parentElement.clientWidth, h=220, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);
      const xmin=-5,xmax=11, pad=30;
      const X = x => pad + (x-xmin)/(xmax-xmin)*(wCss-2*pad);
      const midY = h*0.55;
      ctx.strokeStyle='#1c3441'; ctx.beginPath(); ctx.moveTo(pad,midY); ctx.lineTo(wCss-pad,midY); ctx.stroke();
      // sd band
      ctx.fillStyle='rgba(99,217,196,.13)';
      ctx.fillRect(X(mean-sd), midY-40, X(mean+sd)-X(mean-sd), 80);
      // points
      data.forEach((v,i)=>{
        ctx.beginPath(); ctx.fillStyle='#ffb454';
        ctx.arc(X(v), midY + ((i%2===0)?-1:1)*(10+ (i%5)*6), 3.5, 0, 7); ctx.fill();
      });
      // mean marker
      ctx.strokeStyle='#ff6b6b'; ctx.lineWidth=2;
      ctx.beginPath(); ctx.moveTo(X(mean), midY-55); ctx.lineTo(X(mean), midY+55); ctx.stroke();
      ctx.fillStyle='#ff6b6b'; ctx.font='11px IBM Plex Mono, monospace'; ctx.textAlign='center';
      ctx.fillText('평균 μ', X(mean), midY-62);
      ctx.fillStyle='#63d9c4'; ctx.fillText('±1 표준편차', X(mean), midY+70);
      ctx.textAlign='left';
      rd.innerHTML = `평균 μ = <b>${mean.toFixed(2)}</b> · 분산 σ² = <b>${varr.toFixed(2)}</b> · 표준편차 σ = <b>${sd.toFixed(2)}</b>`;
    }
    window.addEventListener('resize', draw);
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'퍼짐 정도 (분산 조절)', min:0.1,max:2.5,step:0.05,value:spread,
      fmt2:v=>v.toFixed(2), onInput:v=>{ spread=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 평균은 위치(어디에 중심이 있는가), 분산은 모양(얼마나 넓게 퍼져 있는가)을 설명합니다 — 서로 독립적인 정보예요.']));
    root.appendChild(p);
  }
};

const topic4_3 = {
  id:'4-3', title:'4-3강. 균등 분포와 정규 분포',
  desc:`균등분포(Uniform)는 <b>구간 내 모든 값이 동일한 확률</b>을 갖고, 정규분포(Normal/Gaussian)는
  <b>평균 근처에 확률이 몰리고 양 끝으로 갈수록 줄어드는</b> 종 모양입니다. 딥러닝에서 가중치 초기화,
  노이즈 모델링 등에 정규분포가 특히 자주 사용됩니다.`,
  render(root){
    const p = panel('확률밀도함수(PDF) 비교');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let mode='normal', mu=0, sigma=1, a=-2, b=2;
    let graph;
    function normalPdf(x){ return (1/(sigma*Math.sqrt(2*Math.PI))) * Math.exp(-((x-mu)**2)/(2*sigma*sigma)); }
    function uniformPdf(x){ return (x>=a && x<=b) ? 1/(b-a) : 0; }
    function draw(){
      graph.clear(); graph.axes();
      if(mode==='normal'){
        graph.plot(normalPdf, {color:'#ffb454', width:2.2});
        graph.vband(mu-sigma,mu+sigma,'rgba(255,180,84,.12)');
      } else {
        graph.plot(uniformPdf, {color:'#63d9c4', width:2.2});
        graph.vband(a,b,'rgba(99,217,196,.12)');
      }
      rd.innerHTML = mode==='normal'
        ? `N(μ=${mu.toFixed(1)}, σ=${sigma.toFixed(2)}) · 음영: μ±σ 구간 (약 68% 확률질량)`
        : `Uniform(a=${a.toFixed(1)}, b=${b.toFixed(1)}) · 높이 = 1/(b-a) = ${(1/(b-a)).toFixed(3)}`;
    }
    graph = new Graph(canvas,{xmin:-6,xmax:6,ymin:-0.05,ymax:1.1,heightCss:290});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});

    const controlsNormal = el('div',{class:'controls'});
    const muS = makeSlider({label:'μ (평균)', min:-4,max:4,step:0.1,value:mu, fmt2:v=>v.toFixed(1), onInput:v=>{mu=v; draw();}});
    const sigS = makeSlider({label:'σ (표준편차)', min:0.2,max:2.5,step:0.05,value:sigma, fmt2:v=>v.toFixed(2), onInput:v=>{sigma=v; draw();}});
    controlsNormal.appendChild(muS.wrap); controlsNormal.appendChild(sigS.wrap);

    const controlsUniform = el('div',{class:'controls'});
    const aS = makeSlider({label:'a (하한)', min:-5,max:0,step:0.1,value:a, fmt2:v=>v.toFixed(1), onInput:v=>{a=Math.min(v,b-0.2); draw();}});
    const bS = makeSlider({label:'b (상한)', min:0,max:5,step:0.1,value:b, fmt2:v=>v.toFixed(1), onInput:v=>{b=Math.max(v,a+0.2); draw();}});
    controlsUniform.appendChild(aS.wrap); controlsUniform.appendChild(bS.wrap);
    controlsUniform.style.display='none';

    const modeSelect = makeSelect({label:'분포 종류', value:mode,
      options:[{value:'normal',label:'정규분포 (Normal)'},{value:'uniform',label:'균등분포 (Uniform)'}],
      onChange:v=>{ mode=v; controlsNormal.style.display = v==='normal'?'flex':'none'; controlsUniform.style.display = v==='uniform'?'flex':'none'; draw(); }});
    p.appendChild(modeSelect.wrap);
    p.appendChild(controlsNormal); p.appendChild(controlsUniform);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    root.appendChild(p);
  }
};

const CHAPTER_4 = { id:'4', label:'4강 · 확률과 통계', topics:[topic4_1, topic4_2, topic4_3] };

/* =========================================================
   CHAPTER 5 — 순전파/역전파
   ========================================================= */
const topic5_1 = {
  id:'5-1', title:'5-1강. Linear Activation과 인공신경망의 통찰',
  desc:`만약 모든 층에서 활성화함수로 <b>선형함수(linear activation)</b>만 사용한다면, 아무리 층을
  깊게 쌓아도 결국 <b>하나의 선형변환과 수학적으로 동일</b>해집니다 — 비선형 활성화함수가 왜 필수적인지
  보여주는 핵심 통찰입니다. 두 층의 가중치를 조절하며 "등가 단일층"이 항상 존재함을 확인해보세요.`,
  render(root){
    const p = panel('2층 선형 네트워크 = 1층 선형 네트워크');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let w1=1.4, w2=0.7, useNonlinear=false;
    let graph;
    function draw(){
      graph.clear(); graph.axes();
      const layer1 = x=> w1*x;
      const twoLayer = x => useNonlinear ? w2*tanh(layer1(x)) : w2*layer1(x);
      graph.plot(twoLayer, {color:'#ffb454', width:3});
      if(!useNonlinear){
        const wEq = w1*w2;
        graph.plot(x=> wEq*x, {color:'#63d9c4', width:1.4, dash:[6,4]});
        rd.innerHTML = `2층: y = w2·(w1·x) = ${w2.toFixed(2)}×${w1.toFixed(2)}×x  →  등가 1층: y = <b>${wEq.toFixed(2)}</b>·x (완전히 겹쳐 보입니다!)`;
      } else {
        rd.innerHTML = `비선형 활성화(tanh)를 하나라도 넣으면, 더 이상 하나의 직선(1층)으로 표현할 수 없는 <b>곡선</b>이 됩니다.`;
      }
    }
    graph = new Graph(canvas,{xmin:-3,xmax:3,ymin:-4,ymax:4,heightCss:290});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'w1 (1층 가중치)', min:-2,max:2,step:0.05,value:w1, fmt2:v=>v.toFixed(2),
      onInput:v=>{ w1=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'w2 (2층 가중치)', min:-2,max:2,step:0.05,value:w2, fmt2:v=>v.toFixed(2),
      onInput:v=>{ w2=v; draw(); }}).wrap);
    p.appendChild(controls);
    const btnRow=el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>{ useNonlinear=!useNonlinear; draw(); }},['🔀 중간에 tanh 활성화 넣기/빼기']));
    p.appendChild(btnRow);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    root.appendChild(p);
  }
};

const topic5_2 = {
  id:'5-2', title:'5-2강. 역전파 (Backpropagation)',
  desc:`역전파는 <b>연쇄법칙을 반복 적용</b>해서 손실함수 L을 각 가중치로 미분한 값(그래디언트)을
  출력층에서부터 입력층 방향으로 거꾸로 계산하는 알고리즘입니다. 아래는 x → (w1,b1) → sigmoid →
  (w2,b2) → sigmoid → 손실(MSE) 로 이어지는 최소 네트워크의 순전파/역전파 값을 모두 보여줍니다.`,
  render(root){
    const p = panel('미니 네트워크: x → z1 → a1 → z2 → ŷ → Loss');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let x=0.5, w1=0.8, b1=0.1, w2=1.1, b2=-0.2, target=0.9;

    function compute(){
      const z1 = w1*x + b1;
      const a1 = sigmoid(z1);
      const z2 = w2*a1 + b2;
      const yhat = sigmoid(z2);
      const loss = 0.5*(yhat-target)**2;
      // backward
      const dL_dyhat = (yhat-target);
      const dyhat_dz2 = dsigmoid(z2);
      const dL_dz2 = dL_dyhat*dyhat_dz2;
      const dL_dw2 = dL_dz2*a1;
      const dL_db2 = dL_dz2;
      const dL_da1 = dL_dz2*w2;
      const da1_dz1 = dsigmoid(z1);
      const dL_dz1 = dL_da1*da1_dz1;
      const dL_dw1 = dL_dz1*x;
      const dL_db1 = dL_dz1;
      return {z1,a1,z2,yhat,loss,dL_dyhat,dL_dz2,dL_dw2,dL_db2,dL_da1,dL_dz1,dL_dw1,dL_db1};
    }
    function draw(){
      const o = compute();
      fwdOut.innerHTML =
        `<b>순전파</b>&nbsp; z1=w1x+b1=${o.z1.toFixed(3)} → a1=σ(z1)=${o.a1.toFixed(3)} → `+
        `z2=w2a1+b2=${o.z2.toFixed(3)} → ŷ=σ(z2)=${o.yhat.toFixed(3)} → Loss=½(ŷ-y)²=${o.loss.toFixed(4)}`;
      bwdOut.innerHTML =
        `<b>역전파(연쇄법칙)</b>&nbsp; ∂L/∂ŷ=${o.dL_dyhat.toFixed(3)} → ∂L/∂z2=${o.dL_dz2.toFixed(3)} → `+
        `<b>∂L/∂w2=${o.dL_dw2.toFixed(3)}</b>, ∂L/∂b2=${o.dL_db2.toFixed(3)} → ∂L/∂a1=${o.dL_da1.toFixed(3)} → `+
        `∂L/∂z1=${o.dL_dz1.toFixed(3)} → <b>∂L/∂w1=${o.dL_dw1.toFixed(3)}</b>, ∂L/∂b1=${o.dL_db1.toFixed(3)}`;
    }
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'입력 x', min:-2,max:2,step:0.05,value:x, fmt2:v=>v.toFixed(2), onInput:v=>{x=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'목표값 y (target)', min:0,max:1,step:0.02,value:target, fmt2:v=>v.toFixed(2), onInput:v=>{target=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'w1', min:-2,max:2,step:0.05,value:w1, fmt2:v=>v.toFixed(2), onInput:v=>{w1=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'w2', min:-2,max:2,step:0.05,value:w2, fmt2:v=>v.toFixed(2), onInput:v=>{w2=v;draw();}}).wrap);
    p.appendChild(controls);
    const fwdOut = el('div',{class:'readout', style:'flex-direction:column; gap:6px;'});
    const bwdOut = el('div',{class:'readout', style:'flex-direction:column; gap:6px; color:#ffb454;'});
    p.appendChild(fwdOut); p.appendChild(bwdOut);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 굵게 표시된 ∂L/∂w1, ∂L/∂w2가 실제 경사하강법 업데이트에 쓰이는 그래디언트입니다. 역전파는 이 값들을 출력층 쪽에서부터 연쇄법칙으로 "거슬러 올라가며" 계산해요.']));
    root.appendChild(p);
  }
};

const CHAPTER_5 = { id:'5', label:'5강 · 순전파와 역전파', topics:[topic5_1, topic5_2] };

/* =========================================================
   CHAPTER 6 — 분류
   ========================================================= */
const topic6_1 = {
  id:'6-1', title:'6-1강. 왜 이진 분류에서 sigmoid를 사용할까?',
  desc:`Sigmoid 함수 <code>σ(x) = 1/(1+e⁻ˣ)</code>는 실수 전체를 <b>(0,1) 구간으로 압축</b>합니다.
  이 성질 덕분에 출력을 "확률"로 해석할 수 있어서 이진 분류(binary classification)에 자연스럽게
  사용됩니다. 기울기 파라미터 k를 조절하며 "결정 경계가 얼마나 날카로워지는지" 확인해보세요.`,
  render(root){
    const p = panel('σ(k·x) — steepness에 따른 sigmoid 모양');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let k=1.5, x=0.5;
    let graph;
    function draw(){
      graph.clear(); graph.axes();
      graph.hband(0,1,'rgba(99,217,196,.06)');
      graph.plot(v=> sigmoid(k*v), {color:'#ffb454', width:2.2});
      graph.line(-6,0.5,6,0.5,{color:'#5d7683', width:1, dash:[3,3]});
      graph.point(x, sigmoid(k*x), {color:'#63d9c4', r:5});
      rd.innerHTML = `σ(${k.toFixed(1)}×${x.toFixed(2)}) = <b>${sigmoid(k*x).toFixed(3)}</b> → ${sigmoid(k*x)>0.5?'클래스 1로 분류':'클래스 0으로 분류'} (경계값 0.5 기준)`;
    }
    graph = new Graph(canvas,{xmin:-6,xmax:6,ymin:-0.15,ymax:1.15,heightCss:290});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'k (기울기 파라미터)', min:0.2,max:8,step:0.1,value:k, fmt2:v=>v.toFixed(1),
      onInput:v=>{ k=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'x (입력값)', min:-6,max:6,step:0.1,value:x, fmt2:v=>v.toFixed(1),
      onInput:v=>{ x=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 k가 커질수록 sigmoid가 계단함수(step function)에 가까워집니다 — "부드러운 스위치"인 셈이에요.']));
    root.appendChild(p);
  }
};

const topic6_2 = {
  id:'6-2', title:'6-2강. 로지스틱 회귀 (Logistic Regression) · 이진 분류',
  desc:`로지스틱 회귀는 <code>ŷ = σ(wx+b)</code> 형태로 <b>결정 경계(decision boundary)</b>를 학습해서
  두 클래스를 분류합니다. w, b를 조절하며 경계선이 어떻게 데이터를 나누는지, 그리고 색으로 표현된
  확률(σ값)이 어떻게 변하는지 확인해보세요.`,
  render(root){
    const p = panel('2D 산점도 위의 로지스틱 회귀 결정 경계');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const rng = makeRng(21);
    const pts=[];
    for(let i=0;i<14;i++) pts.push({x: 1.5+gaussian(rng)*0.9, y: 1.5+gaussian(rng)*0.9, c:0});
    for(let i=0;i<14;i++) pts.push({x: -1.5+gaussian(rng)*0.9, y: -1.5+gaussian(rng)*0.9, c:1});
    let w1=1, w2=1, b=0;
    let graph;
    function draw(){
      graph.clear();
      // background probability field
      const ctx=graph.ctx, p_=graph.pad;
      for(let px=p_; px<graph.w-p_; px+=4){
        for(let py=p_; py<graph.h-p_; py+=4){
          const x = graph.xmin + (px-p_)/(graph.w-2*p_)*(graph.xmax-graph.xmin);
          const y = graph.ymin + (graph.h-p_-py)/(graph.h-2*p_)*(graph.ymax-graph.ymin);
          const s = sigmoid(w1*x+w2*y+b);
          const [r,g,bl] = rampColor(s);
          ctx.fillStyle = `rgba(${r},${g},${bl},0.35)`;
          ctx.fillRect(px,py,4,4);
        }
      }
      graph.axes();
      pts.forEach(pt=> graph.point(pt.x,pt.y,{color: pt.c===0?'#ffb454':'#63d9c4', r:4.5}));
      // decision boundary line w1 x + w2 y + b = 0  ->  y = -(w1 x + b)/w2
      if(Math.abs(w2)>0.01){
        graph.plot(x=> -(w1*x+b)/w2, {color:'#e7f0f2', width:2});
      }
      let correct=0;
      pts.forEach(pt=>{ const pred = sigmoid(w1*pt.x+w2*pt.y+b)>0.5?0:1; if(pred===pt.c) correct++; });
      rd.innerHTML = `결정 경계: w1x+w2y+b=0 · 정확도: <b>${correct}/${pts.length}</b>`;
    }
    graph = new Graph(canvas,{xmin:-4,xmax:4,ymin:-4,ymax:4,heightCss:320});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'w1', min:-3,max:3,step:0.1,value:w1, fmt2:v=>v.toFixed(1), onInput:v=>{w1=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'w2', min:-3,max:3,step:0.1,value:w2, fmt2:v=>v.toFixed(1), onInput:v=>{w2=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'b', min:-3,max:3,step:0.1,value:b, fmt2:v=>v.toFixed(1), onInput:v=>{b=v;draw();}}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'legend'},[
      el('span',{},[el('i',{style:'background:#ffb454'}),'클래스 0']),
      el('span',{},[el('i',{style:'background:#63d9c4'}),'클래스 1']),
      el('span',{},[el('i',{style:'background:#e7f0f2'}),'결정 경계선']),
    ]));
    root.appendChild(p);
  }
};

const topic6_4 = {
  id:'6-4', title:'6-4강. 교차 엔트로피(Cross-Entropy)와 MLE',
  desc:`로지스틱 회귀의 손실함수로 MSE 대신 <b>교차 엔트로피</b>를 쓰는 이유는, 확률적으로 정답에서
  틀릴수록(예측이 확신에 차서 틀릴수록) <b>훨씬 더 크게 벌점</b>을 주기 때문입니다: 
  <code>L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]</code>. 이는 통계학의 <b>최대우도추정(MLE)</b> 관점과도
  일치합니다. 예측확률을 슬라이더로 움직이며 손실이 어떻게 폭증하는지 보세요.`,
  render(root){
    const p = panel('Cross-Entropy Loss vs 예측확률 ŷ');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let yhat=0.7, label=1;
    let graph;
    function ce(y, yh){ yh=clamp(yh,1e-6,1-1e-6); return -(y*Math.log(yh) + (1-y)*Math.log(1-yh)); }
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(v=> ce(label, v), {color:'#ffb454', width:2.2});
      graph.point(yhat, ce(label,yhat), {color:'#63d9c4', r:5.5});
      rd.innerHTML = `정답 라벨 y=<b>${label}</b>, 예측 ŷ=<b>${yhat.toFixed(2)}</b> → Loss = <b>${ce(label,yhat).toFixed(3)}</b> ${yhat<0.05&&label===1?' (거의 무한대로 폭증!)':''}`;
    }
    graph = new Graph(canvas,{xmin:0,xmax:1,ymin:0,ymax:5,heightCss:290});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'예측확률 ŷ', min:0.01,max:0.99,step:0.01,value:yhat, fmt2:v=>v.toFixed(2),
      onInput:v=>{ yhat=v; draw(); }}).wrap);
    controls.appendChild(makeSelect({label:'정답 라벨 y', value:String(label),
      options:[{value:'1',label:'1 (Positive)'},{value:'0',label:'0 (Negative)'}],
      onChange:v=>{ label=parseInt(v); draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 정답이 1인데 모델이 확신을 갖고 0에 가깝게 예측하면(ŷ→0) 손실이 무한대로 치솟습니다 — "확신에 찬 오답"에 강한 페널티를 주는 게 교차 엔트로피의 핵심이에요.']));
    root.appendChild(p);
  }
};

const topic6_5 = {
  id:'6-5', title:'6-5강. 소프트맥스 회귀 (Softmax Regression) · 다중 분류',
  desc:`클래스가 3개 이상일 때는 <b>소프트맥스 함수</b>로 각 클래스의 점수(logit)를 확률로 변환합니다:
  <code>softmax(z)ᵢ = eᶻⁱ / Σⱼeᶻʲ</code>. 모든 확률의 합은 항상 1입니다. 세 클래스의 logit을 조절하며
  확률 막대가 어떻게 재분배되는지 확인해보세요.`,
  render(root){
    const p = panel('3-클래스 Softmax');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let z=[1.0,0.2,-0.5];
    const labels=['고양이','강아지','토끼'];
    const colors=['#ffb454','#63d9c4','#b48bf2'];
    function softmax(z){
      const m = Math.max(...z);
      const exps = z.map(v=>Math.exp(v-m));
      const s = exps.reduce((a,b)=>a+b,0);
      return exps.map(e=>e/s);
    }
    function draw(){
      const probs = softmax(z);
      const wCss = canvas.parentElement.clientWidth, h=260, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);
      const n=z.length, barW=Math.min(90,wCss/(n*2));
      probs.forEach((pr,i)=>{
        const cx = wCss*(i+0.5)/n;
        const barH = pr*(h-70);
        ctx.fillStyle=colors[i];
        ctx.fillRect(cx-barW/2, h-40-barH, barW, barH);
        ctx.fillStyle='#e7f0f2'; ctx.font='12px IBM Plex Mono, monospace'; ctx.textAlign='center';
        ctx.fillText(labels[i], cx, h-18);
        ctx.fillStyle=colors[i]; ctx.font='13px IBM Plex Mono, monospace';
        ctx.fillText((pr*100).toFixed(1)+'%', cx, h-46-barH);
        ctx.fillStyle='#93aab5'; ctx.font='10px IBM Plex Mono, monospace';
        ctx.fillText('z='+z[i].toFixed(2), cx, h-4);
        ctx.textAlign='left';
      });
      rd.innerHTML = `확률 합계: <b>${probs.reduce((a,b)=>a+b,0).toFixed(4)}</b> (항상 1) · 가장 높은 확률: <b>${labels[probs.indexOf(Math.max(...probs))]}</b>`;
    }
    window.addEventListener('resize', draw);
    const controls = el('div',{class:'controls'});
    z.forEach((v,i)=>{
      controls.appendChild(makeSlider({label:`logit z (${labels[i]})`, min:-3,max:3,step:0.1,value:v, fmt2:vv=>vv.toFixed(1),
        onInput:vv=>{ z[i]=vv; draw(); }}).wrap);
    });
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    root.appendChild(p);
  }
};

const CHAPTER_6 = { id:'6', label:'6강 · 분류(Classification)', topics:[topic6_1, topic6_2, topic6_4, topic6_5] };

/* =========================================================
   CHAPTER 7 — 보편 근사 정리
   ========================================================= */
const topic7 = {
  id:'7', title:'7강. Universal Approximation Theorem (보편 근사 정리)',
  desc:`은닉층이 하나뿐인 신경망이라도, <b>충분히 많은 은닉 뉴런</b>이 있다면 임의의 연속함수를
  원하는 정확도로 근사할 수 있다는 정리입니다. 여기서는 sigmoid 뉴런들을 계단처럼 쌓아
  목표 함수를 근사하는 방식을 보여줍니다 — 은닉 뉴런 개수(n)를 늘려보세요.`,
  render(root){
    const p = panel('sigmoid 뉴런 n개로 목표 함수 근사하기');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let n=4;
    const target = x => 0.4*Math.sin(2.2*x) + 0.25*Math.cos(0.8*x) + 0.3;
    let graph;
    function approx(x, n){
      const xmin=-4, xmax=4;
      let acc = target(xmin);
      const steep = 25;
      for(let i=0;i<n;i++){
        const x0 = lerp(xmin,xmax,i/n);
        const x1 = lerp(xmin,xmax,(i+1)/n);
        const jump = target(x1)-target(x0);
        acc += jump * sigmoid(steep*(x-x0));
      }
      return acc;
    }
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(target, {color:'#63d9c4', width:2.2});
      graph.plot(x=> approx(x,n), {color:'#ffb454', width:2, dash:[5,3]});
      let err=0, cnt=0;
      for(let x=-4;x<=4;x+=0.1){ err += Math.abs(target(x)-approx(x,n)); cnt++; }
      rd.innerHTML = `은닉 뉴런 개수 n = <b>${n}</b> · 평균 근사 오차 = <b>${(err/cnt).toFixed(4)}</b>`;
    }
    graph = new Graph(canvas,{xmin:-4,xmax:4,ymin:-0.6,ymax:1.3,heightCss:300});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'은닉 뉴런 개수 n', min:1,max:60,step:1,value:n, fmt2:v=>v,
      onInput:v=>{ n=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'legend'},[
      el('span',{},[el('i',{style:'background:#63d9c4'}),'목표 함수']),
      el('span',{},[el('i',{style:'background:#ffb454'}),'sigmoid n개의 합으로 근사']),
    ]));
    p.appendChild(el('div',{class:'note'},['💡 n을 늘릴수록 주황색 근사 곡선이 청록색 목표 함수에 거의 완벽히 겹쳐집니다 — "뉴런이 충분히 많으면 뭐든 근사할 수 있다"는 것이 이 정리의 직관이에요.']));
    root.appendChild(p);
  }
};
const CHAPTER_7 = { id:'7', label:'7강 · 보편 근사 정리', topics:[topic7] };

/* =========================================================
   CHAPTER 8 — 학습 안정화와 일반화
   ========================================================= */
const topic8_1 = {
  id:'8-1', title:'8-1강. Vanishing Gradient(기울기 소실)와 ReLU',
  desc:`역전파는 각 층의 미분값을 <b>곱해나가며</b> 앞쪽 층까지 그래디언트를 전달합니다.
  sigmoid/tanh의 미분값은 최대가 각각 0.25, 1.0으로 작아서, 층이 깊어질수록 그래디언트가
  <b>0에 가깝게 소실</b>됩니다. ReLU는 양수 구간에서 미분이 항상 1이라 이 문제를 크게 줄여줍니다.
  깊이(층 수)를 늘려가며 그래디언트 크기가 어떻게 변하는지 관찰해보세요.`,
  render(root){
    const p = panel('층 깊이에 따른 그래디언트 크기 (전형적인 활성값 x=0.5 기준)');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let depth=10, actName='sigmoid';
    const derivs = { sigmoid: dsigmoid, tanh: dtanh, relu: drelu };
    let graph;
    function gradAtLayer(l){
      const d = derivs[actName];
      const perLayerDeriv = d(0.5) * 1.0; // typical weight ~1
      return Math.pow(perLayerDeriv, l);
    }
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(l=> gradAtLayer(Math.round(l)), {color:'#ffb454', width:2});
      for(let l=0;l<=depth;l++){
        graph.point(l, gradAtLayer(l), {color:'#63d9c4', r:3});
      }
      rd.innerHTML = `${depth}층 통과 후 그래디언트 크기: <b>${gradAtLayer(depth).toExponential(3)}</b> (활성화: ${actName}, 층당 미분값: ${derivs[actName](0.5).toFixed(3)})`;
    }
    graph = new Graph(canvas,{xmin:0,xmax:20,ymin:0,ymax:1.05,heightCss:290});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSelect({label:'활성화함수', value:actName,
      options:[{value:'sigmoid',label:'Sigmoid (최대 미분 0.25)'},{value:'tanh',label:'Tanh (최대 미분 1.0)'},{value:'relu',label:'ReLU (양수 구간 미분 1.0)'}],
      onChange:v=>{ actName=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'층 깊이', min:1,max:20,step:1,value:depth, fmt2:v=>v,
      onInput:v=>{ depth=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 sigmoid를 20층 쌓으면 그래디언트가 사실상 0이 되어 앞쪽 층은 거의 학습되지 않아요. ReLU는 미분이 1이라 곱해도 크기가 줄지 않는다는 게 핵심 장점입니다 (다만 입력이 음수면 미분이 0이 되는 "죽은 뉴런" 문제는 남아있어요).']));
    root.appendChild(p);
  }
};

const topic8_2 = {
  id:'8-2', title:'8-2강. Batch Normalization & Layer Normalization',
  desc:`층을 통과하며 활성화 값의 분포가 계속 바뀌는 <b>내부 공변량 변화(internal covariate shift)</b>는
  학습을 불안정하게 만듭니다. 정규화는 배치의 평균/분산으로 값을 표준화한 뒤(<code>(x-μ)/σ</code>),
  학습 가능한 <b>γ(scale), β(shift)</b>로 다시 조정합니다. 아래에서 임의로 치우친 활성화 분포가
  정규화 후 어떻게 안정되는지 확인해보세요.`,
  render(root){
    const p = panel('정규화 전 / 후 활성화 분포');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let shift=3.5, scaleIn=2.2, gamma=1.0, beta=0.0;
    const rng = makeRng(9);
    const base = Array.from({length:200},()=> gaussian(rng));
    function draw(){
      const raw = base.map(z=> shift + z*scaleIn);
      const mean = raw.reduce((a,b)=>a+b,0)/raw.length;
      const varr = raw.reduce((a,b)=>a+(b-mean)**2,0)/raw.length;
      const sd = Math.sqrt(varr)||1e-6;
      const normed = raw.map(v=> gamma*((v-mean)/sd) + beta);

      const wCss = canvas.parentElement.clientWidth, h=280, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);

      function hist(data, x0, colW, color, tag){
        const bins=24, xmin=-8, xmax=8;
        const hcount=Array(bins).fill(0);
        data.forEach(v=>{ const idx=clamp(Math.floor((v-xmin)/(xmax-xmin)*bins),0,bins-1); hcount[idx]++; });
        const maxC = Math.max(...hcount,1);
        ctx.strokeStyle='#1c3441'; ctx.strokeRect(x0+8,10,colW-16,h-46);
        hcount.forEach((c,i)=>{
          const bh=(c/maxC)*(h-70);
          ctx.fillStyle=color;
          ctx.fillRect(x0+10+i*((colW-20)/bins), h-36-bh, (colW-20)/bins-1, bh);
        });
        ctx.fillStyle='#93aab5'; ctx.font='11px IBM Plex Mono, monospace'; ctx.textAlign='center';
        ctx.fillText(tag, x0+colW/2, h-16);
        ctx.textAlign='left';
      }
      hist(raw, 0, wCss/2, '#ff6b6b', '정규화 전');
      hist(normed, wCss/2, wCss/2, '#63d9c4', '정규화 후 (γ,β 적용)');
      rd.innerHTML = `정규화 전: μ=${mean.toFixed(2)}, σ=${sd.toFixed(2)} → 정규화 후: μ≈${beta.toFixed(2)}, σ≈${Math.abs(gamma).toFixed(2)} 로 재조정`;
    }
    window.addEventListener('resize', draw);
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'입력 분포 shift (치우침)', min:-6,max:6,step:0.1,value:shift, fmt2:v=>v.toFixed(1), onInput:v=>{shift=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'입력 분포 scale (퍼짐)', min:0.3,max:5,step:0.1,value:scaleIn, fmt2:v=>v.toFixed(1), onInput:v=>{scaleIn=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'γ (학습된 scale)', min:0.1,max:3,step:0.1,value:gamma, fmt2:v=>v.toFixed(1), onInput:v=>{gamma=v;draw();}}).wrap);
    controls.appendChild(makeSlider({label:'β (학습된 shift)', min:-4,max:4,step:0.1,value:beta, fmt2:v=>v.toFixed(1), onInput:v=>{beta=v;draw();}}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 Batch Norm은 "배치(여러 샘플)" 단위로, Layer Norm은 "한 샘플의 여러 뉴런" 단위로 평균/분산을 계산한다는 차이가 있어요 — 정규화 자체의 원리는 동일합니다.']));
    root.appendChild(p);
  }
};

const topic8_3 = {
  id:'8-3', title:'8-3강. Loss Landscape이 꼬불꼬불해진다?!',
  desc:`실제 신경망의 손실함수는 2-4강에서 본 매끈한 그릇 모양이 아니라, <b>여러 개의 국소 최솟값
  (local minima)과 안장점(saddle point)</b>이 있는 울퉁불퉁한 지형입니다. 시작점을 바꿔가며
  경사하강법이 전역 최솟값이 아닌 다른 골짜기에 갇힐 수 있다는 걸 확인해보세요.`,
  render(root){
    const p = panel('울퉁불퉁한 손실 지형 위의 경사하강');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const f = (x,y)=> 0.35*(x*x+y*y) + Math.sin(x*2.2)*Math.cos(y*2.2)*1.4;
    const gradf = (x,y,h=1e-3)=> [ (f(x+h,y)-f(x-h,y))/(2*h), (f(x,y+h)-f(x,y-h))/(2*h) ];
    let x0=2.6, y0=1.8, lr=0.05, iters=40;
    let path=[];
    const wCanvas = document.createElement('canvas');
    function heat(){
      const wCss=420, res=90;
      wCanvas.width=res; wCanvas.height=res;
      const ictx = wCanvas.getContext('2d');
      const img = ictx.createImageData(res,res);
      let vmin=Infinity,vmax=-Infinity;
      const vals=[];
      for(let j=0;j<res;j++){ vals.push([]); for(let i=0;i<res;i++){
        const x = lerp(-4,4,i/(res-1)), y = lerp(4,-4,j/(res-1));
        const v = f(x,y); vals[j].push(v); vmin=Math.min(vmin,v); vmax=Math.max(vmax,v);
      }}
      for(let j=0;j<res;j++) for(let i=0;i<res;i++){
        const t = (vals[j][i]-vmin)/((vmax-vmin)||1);
        const [r,g,b]=rampColor(t);
        const idx=(j*res+i)*4;
        img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
      }
      ictx.putImageData(img,0,0);
    }
    heat();
    function compute(){
      path=[[x0,y0]]; let x=x0,y=y0;
      for(let i=0;i<iters;i++){ const [gx,gy]=gradf(x,y); x-=lr*gx; y-=lr*gy; path.push([x,y]); }
    }
    let graph;
    function draw(){
      const ctx = graph.ctx;
      ctx.imageSmoothingEnabled=false;
      ctx.drawImage(wCanvas, graph.pad, graph.pad, graph.w-2*graph.pad, graph.h-2*graph.pad);
      graph.axes();
      for(let i=0;i<path.length-1;i++) graph.line(path[i][0],path[i][1],path[i+1][0],path[i+1][1],{color:'#e7f0f2', width:1.6});
      path.forEach((pt,i)=> graph.point(pt[0],pt[1],{color: i===path.length-1?'#ff6b6b':'#e7f0f2', r:i===path.length-1?5.5:2}));
      rd.innerHTML = `시작 손실: <b>${fmt(f(x0,y0),3)}</b> → 도착 손실: <b>${fmt(f(...path.at(-1)),3)}</b> (전역 최솟값이 아닐 수 있어요 — 시작점을 바꿔보세요)`;
    }
    graph = new Graph(canvas,{xmin:-4,xmax:4,ymin:-4,ymax:4,heightCss:320});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'시작 x₀', min:-3.8,max:3.8,step:0.1,value:x0, fmt2:v=>v.toFixed(1), onInput:v=>{x0=v; compute(); draw();}}).wrap);
    controls.appendChild(makeSlider({label:'시작 y₀', min:-3.8,max:3.8,step:0.1,value:y0, fmt2:v=>v.toFixed(1), onInput:v=>{y0=v; compute(); draw();}}).wrap);
    controls.appendChild(makeSlider({label:'학습률', min:0.005,max:0.15,step:0.005,value:lr, fmt2:v=>v.toFixed(3), onInput:v=>{lr=v; compute(); draw();}}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    compute(); draw();
    root.appendChild(p);
  }
};

const topic8_4 = {
  id:'8-4', title:'8-4강. 과적합(Overfitting)과 데이터 증강(Data Augmentation)',
  desc:`모델이 <b>학습 데이터의 노이즈까지 외워버려서</b> 새로운 데이터에 일반화하지 못하는 현상이
  과적합입니다. 다항식 차수(degree)를 높여가며 학습 데이터에는 완벽히 맞지만 진짜 패턴과는
  멀어지는 곡선을 확인하고, "데이터 증강"으로 데이터를 늘리면 과적합이 줄어드는 것도 확인해보세요.`,
  render(root){
    const p = panel('다항 회귀의 과적합');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const rng = makeRng(13);
    const trueF = x => Math.sin(x*1.3);
    let baseXs=[], baseYs=[];
    for(let i=0;i<9;i++){ const x=lerp(-3,3,i/8); baseXs.push(x); baseYs.push(trueF(x)+gaussian(rng)*0.25); }
    let degree=1, augment=false;
    let graph;
    function currentData(){
      if(!augment) return {xs:baseXs, ys:baseYs};
      const xs=[...baseXs], ys=[...baseYs];
      const rng2 = makeRng(77);
      baseXs.forEach((x,i)=>{
        for(let k=0;k<3;k++){ xs.push(x+gaussian(rng2)*0.15); ys.push(baseYs[i]+gaussian(rng2)*0.1); }
      });
      return {xs,ys};
    }
    function draw(){
      graph.clear(); graph.axes();
      graph.plot(trueF, {color:'#3d6b78', width:1.6, dash:[3,3]});
      const {xs,ys} = currentData();
      const deg = Math.min(degree, xs.length-1);
      const coeffs = polyFit(xs,ys,deg);
      graph.plot(x=> polyEval(coeffs,x), {color:'#ffb454', width:2.2});
      xs.forEach((x,i)=> graph.point(x,ys[i],{color:'#63d9c4', r:3}));
      let mse=0; xs.forEach((x,i)=> mse += (ys[i]-polyEval(coeffs,x))**2); mse/=xs.length;
      rd.innerHTML = `차수: <b>${deg}</b> · 데이터 증강: <b>${augment?'ON (샘플 ' + xs.length + '개)':'OFF (샘플 ' + xs.length + '개)'}</b> · 학습 MSE: <b>${mse.toFixed(4)}</b>`;
    }
    graph = new Graph(canvas,{xmin:-3.5,xmax:3.5,ymin:-2.5,ymax:2.5,heightCss:300});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'다항식 차수', min:1,max:14,step:1,value:degree, fmt2:v=>v,
      onInput:v=>{ degree=v; draw(); }}).wrap);
    p.appendChild(controls);
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:()=>{ augment=!augment; draw(); }},['🔁 데이터 증강 켜기/끄기']));
    p.appendChild(btnRow);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'note'},['💡 차수를 10 이상으로 올리면 주황 곡선이 점(학습 데이터)을 거의 완벽히 지나가지만, 점선(진짜 패턴)에서는 심하게 벗어나요 — 이게 과적합입니다. 데이터 증강을 켜면 같은 차수에서도 곡선이 훨씬 안정돼요.']));
    root.appendChild(p);
  }
};

const topic8_5 = {
  id:'8-5', title:'8-5강. Dropout에 대한 진실',
  desc:`학습 중 매 스텝마다 <b>무작위로 일부 뉴런을 꺼버려서(dropout)</b> 특정 뉴런에 과도하게
  의존하지 못하게 만드는 정규화 기법입니다. 결과적으로 여러 개의 서로 다른 "얇은 네트워크"를
  앙상블하는 효과를 냅니다. Dropout 비율을 조절하고 "새 에폭"을 눌러 매번 다른 뉴런이 꺼지는 걸 확인해보세요.`,
  render(root){
    const p = panel('네트워크 다이어그램 위의 Dropout');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    let rate=0.4;
    const layers = [3,6,6,2];
    let active = layers.map(n=>Array(n).fill(true));
    const rng = makeRng(31);
    function newEpoch(){
      active = layers.map((n,li)=> Array(n).fill(0).map(()=> (li===0||li===layers.length-1) ? true : rng()>=rate));
      draw();
    }
    function draw(){
      const wCss = canvas.parentElement.clientWidth, h=300, dpr=window.devicePixelRatio||1;
      canvas.style.width=wCss+'px'; canvas.style.height=h+'px';
      canvas.width=wCss*dpr; canvas.height=h*dpr;
      const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle='#081019'; ctx.fillRect(0,0,wCss,h);
      const gap = wCss/(layers.length+1);
      const positions = layers.map((n,li)=> Array.from({length:n},(_,i)=> [gap*(li+1), h*(i+1)/(n+1)]));
      // edges
      for(let li=0; li<layers.length-1; li++){
        positions[li].forEach((p1,i)=>{
          positions[li+1].forEach((p2,j)=>{
            const on = active[li][i] && active[li+1][j];
            ctx.strokeStyle = on ? 'rgba(255,180,84,.35)' : 'rgba(93,118,131,.08)';
            ctx.lineWidth=1;
            ctx.beginPath(); ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]); ctx.stroke();
          });
        });
      }
      // nodes
      layers.forEach((n,li)=>{
        positions[li].forEach((pos,i)=>{
          const on = active[li][i];
          ctx.beginPath(); ctx.arc(pos[0],pos[1],13,0,7);
          ctx.fillStyle = on ? '#101f2b' : '#0b1620';
          ctx.fill();
          ctx.strokeStyle = on ? '#ffb454' : '#3a4650'; ctx.lineWidth=2; ctx.stroke();
        });
      });
      const hiddenTotal = layers.slice(1,-1).reduce((a,b)=>a+b,0);
      const hiddenOn = active.slice(1,-1).reduce((a,arr)=>a+arr.filter(Boolean).length,0);
      rd.innerHTML = `은닉 뉴런 활성: <b>${hiddenOn}/${hiddenTotal}</b> (dropout rate ${(rate*100).toFixed(0)}%) · 입력/출력층은 항상 유지됩니다`;
    }
    window.addEventListener('resize', draw);
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSlider({label:'Dropout 비율 p', min:0,max:0.8,step:0.05,value:rate, fmt2:v=>Math.round(v*100)+'%',
      onInput:v=>{ rate=v; newEpoch(); }}).wrap);
    p.appendChild(controls);
    const btnRow = el('div',{class:'btn-row'});
    btnRow.appendChild(el('button',{class:'btn', onclick:newEpoch},['🔁 새 에폭 (다시 무작위로 끄기)']));
    p.appendChild(btnRow);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    newEpoch();
    p.appendChild(el('div',{class:'note'},['💡 실제 학습에서는 매 미니배치마다 이런 무작위 마스킹이 반복되고, 테스트 시에는 모든 뉴런을 사용하되 그 비율만큼 출력을 스케일링합니다 (inverted dropout).']));
    root.appendChild(p);
  }
};

const topic8_6 = {
  id:'8-6', title:'8-6강. 정규화(Regularization)와 MAP',
  desc:`손실함수에 <b>가중치 크기에 대한 벌점</b>을 추가해서 모델이 너무 복잡해지는(=과적합) 것을
  막는 기법입니다. L2(<code>λΣw²</code>)는 원형 제약, L1(<code>λΣ|w|</code>)은 다이아몬드 제약을
  거는 것과 같고, 이는 베이즈 통계의 <b>MAP(Maximum A Posteriori)</b> 추정에서 가중치에 사전분포를
  부여하는 것과 수학적으로 동일합니다. λ를 조절하며 최적해가 원점 쪽으로 끌려오는 걸 확인해보세요.`,
  render(root){
    const p = panel('L1 vs L2 정규화가 최적해를 끌어당기는 방식');
    const cw = el('div',{class:'canvas-wrap'});
    const canvas = el('canvas'); cw.appendChild(canvas); p.appendChild(cw);
    const w_ml = [2.4, 1.6]; // unregularized MLE solution
    let lambda=0.3, kind='l2';
    let graph;
    function regularizedSolution(){
      if(kind==='l2'){
        return [w_ml[0]/(1+lambda), w_ml[1]/(1+lambda)];
      } else {
        const soft = v => Math.sign(v)*Math.max(Math.abs(v)-lambda,0);
        return [soft(w_ml[0]), soft(w_ml[1])];
      }
    }
    function draw(){
      graph.clear(); graph.axes();
      // loss contours around w_ml (elliptical, isotropic here for simplicity)
      for(let k=1;k<=6;k++){
        const r = k*0.5;
        graph.ctx.strokeStyle = 'rgba(61,140,150,'+(0.12+k*0.03)+')';
        graph.ctx.lineWidth=1; graph.ctx.beginPath();
        for(let a=0;a<=64;a++){
          const th=a/64*2*Math.PI;
          const x = w_ml[0]+r*Math.cos(th), y = w_ml[1]+r*Math.sin(th)*0.7;
          const px=graph.X(x), py=graph.Y(y);
          if(a===0) graph.ctx.moveTo(px,py); else graph.ctx.lineTo(px,py);
        }
        graph.ctx.stroke();
      }
      // constraint region
      graph.ctx.strokeStyle='#b48bf2'; graph.ctx.lineWidth=2; graph.ctx.beginPath();
      const R = 1/(lambda+0.15)*0.9;
      if(kind==='l2'){
        for(let a=0;a<=64;a++){ const th=a/64*2*Math.PI; const px=graph.X(R*Math.cos(th)), py=graph.Y(R*Math.sin(th)); if(a===0) graph.ctx.moveTo(px,py); else graph.ctx.lineTo(px,py); }
      } else {
        const pts=[[R,0],[0,R],[-R,0],[0,-R],[R,0]];
        pts.forEach((pt,i)=>{ const px=graph.X(pt[0]), py=graph.Y(pt[1]); if(i===0) graph.ctx.moveTo(px,py); else graph.ctx.lineTo(px,py); });
      }
      graph.ctx.stroke();
      graph.point(w_ml[0],w_ml[1],{color:'#ff6b6b', r:5.5});
      const [rw0,rw1] = regularizedSolution();
      graph.point(rw0,rw1,{color:'#ffb454', r:5.5});
      rd.innerHTML = `정규화 없는 최적해(MLE): (<b>${w_ml[0].toFixed(2)}, ${w_ml[1].toFixed(2)}</b>) → 정규화된 해(MAP): (<b>${rw0.toFixed(2)}, ${rw1.toFixed(2)}</b>)`;
    }
    graph = new Graph(canvas,{xmin:-3.5,xmax:3.5,ymin:-3.5,ymax:3.5,heightCss:320});
    window.addEventListener('resize', ()=>{graph.resize(); draw();});
    const controls = el('div',{class:'controls'});
    controls.appendChild(makeSelect({label:'정규화 종류', value:kind,
      options:[{value:'l2',label:'L2 (Ridge, 원형 제약)'},{value:'l1',label:'L1 (Lasso, 다이아몬드 제약)'}],
      onChange:v=>{ kind=v; draw(); }}).wrap);
    controls.appendChild(makeSlider({label:'λ (정규화 강도)', min:0,max:2,step:0.05,value:lambda, fmt2:v=>v.toFixed(2),
      onInput:v=>{ lambda=v; draw(); }}).wrap);
    p.appendChild(controls);
    const rd = el('div',{class:'readout'}); p.appendChild(rd);
    draw();
    p.appendChild(el('div',{class:'legend'},[
      el('span',{},[el('i',{style:'background:#ff6b6b'}),'정규화 없는 해 (과적합 위험)']),
      el('span',{},[el('i',{style:'background:#ffb454'}),'정규화된 해 (원점 쪽으로 수축)']),
      el('span',{},[el('i',{style:'background:#b48bf2'}),'제약 영역 경계']),
    ]));
    p.appendChild(el('div',{class:'note'},['💡 L1은 축 위(즉 한 가중치가 정확히 0)에서 만날 가능성이 높아 <b>희소성(sparsity)</b>을 만들고, L2는 모든 가중치를 고르게 작게 줄여요. λ를 키울수록 amber 점이 원점 쪽으로 더 강하게 끌려옵니다.']));
    root.appendChild(p);
  }
};

const CHAPTER_8 = { id:'8', label:'8강 · 학습 안정화와 일반화', topics:[topic8_1, topic8_2, topic8_3, topic8_4, topic8_5, topic8_6] };

/* =========================================================
   APP DRIVER — navigation, routing, scope-strip animation
   ========================================================= */
const CHAPTERS = [CHAPTER_1, CHAPTER_2, CHAPTER_3, CHAPTER_4, CHAPTER_5, CHAPTER_6, CHAPTER_7, CHAPTER_8];

function buildSidebar(){
  const nav = document.getElementById('chapterNav');
  nav.innerHTML = '';
  CHAPTERS.forEach((ch, ci) => {
    const group = el('div',{class:'chapter-group'+(ci===0?' open':'')});
    const head = el('button',{class:'chapter-head'}, [
      el('span',{}, [el('span',{class:'chnum'},[ch.id]), ch.label.split('· ')[1] || ch.label]),
      el('span',{class:'chev'}, ['▸']),
    ]);
    head.addEventListener('click', ()=> group.classList.toggle('open'));
    const list = el('div',{class:'topic-list'});
    ch.topics.forEach(t=>{
      const btn = el('button',{class:'topic-btn', 'data-id':t.id}, [t.title]);
      btn.addEventListener('click', ()=> goToTopic(t.id));
      list.appendChild(btn);
    });
    group.appendChild(head); group.appendChild(list);
    nav.appendChild(group);
  });
}

function findTopic(id){
  for(const ch of CHAPTERS){
    const t = ch.topics.find(t=>t.id===id);
    if(t) return {topic:t, chapter:ch};
  }
  return null;
}

function flatTopicList(){
  return CHAPTERS.flatMap(ch=>ch.topics);
}

function goToTopic(id){
  const found = findTopic(id);
  if(!found) return;
  const {topic, chapter} = found;
  const stage = document.getElementById('stage');
  stage.innerHTML = '';

  stage.appendChild(el('div',{class:'topic-eyebrow'}, [chapter.label]));
  stage.appendChild(el('h1',{class:'topic-title'}, [topic.title]));
  stage.appendChild(el('div',{class:'topic-desc', html: topic.desc}));

  try{
    topic.render(stage);
  }catch(err){
    stage.appendChild(el('div',{class:'panel'}, ['이 데모를 불러오는 중 오류가 발생했습니다: '+err.message]));
    console.error(err);
  }

  // prev/next
  const flat = flatTopicList();
  const idx = flat.findIndex(t=>t.id===id);
  const prev = flat[idx-1], next = flat[idx+1];
  const nav = el('div',{class:'prevnext'});
  nav.appendChild(prev ? el('button',{class:'btn', onclick:()=>goToTopic(prev.id)},['← '+prev.title]) : el('span',{}));
  nav.appendChild(next ? el('button',{class:'btn', onclick:()=>goToTopic(next.id)},[next.title+' →']) : el('span',{}));
  stage.appendChild(nav);

  // active state in sidebar
  document.querySelectorAll('.topic-btn').forEach(b=> b.classList.toggle('active', b.getAttribute('data-id')===id));
  // ensure the containing group is open
  const activeBtn = document.querySelector('.topic-btn.active');
  if(activeBtn){
    const group = activeBtn.closest('.chapter-group');
    if(group) group.classList.add('open');
  }
  window.scrollTo({top:0, behavior:'smooth'});
  document.getElementById('sidebar').classList.remove('open');

  // update scope strip label
  document.getElementById('scopeText').textContent = 'SIGNAL: ' + chapter.label + ' · ' + topic.id;
  scopeMode = chapterScopeMode(chapter.id);

  history.replaceState(null,'', '#'+id);
}

function chapterScopeMode(chId){
  // different waveform "personality" per chapter — purely decorative signature element
  return {
    '1':'sine', '2':'sine2', '3':'chirp', '4':'noise', '5':'square', '6':'step', '7':'sumSine', '8':'noisyDecay'
  }[chId] || 'sine';
}

/* ---- oscilloscope header animation (signature visual element) ---- */
let scopeMode = 'sine';
function initScope(){
  const canvas = document.getElementById('scopeCanvas');
  const ctx = canvas.getContext('2d');
  function resize(){
    const dpr = window.devicePixelRatio||1;
    const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
    canvas.width = w*dpr; canvas.height = h*dpr;
    canvas.style.width=w+'px'; canvas.style.height=h+'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  resize(); window.addEventListener('resize', resize);
  let t = 0;
  function waveAt(x, phase){
    switch(scopeMode){
      case 'sine': return Math.sin(x*0.05 + phase);
      case 'sine2': return Math.sin(x*0.05+phase)*0.6 + Math.sin(x*0.11+phase*1.7)*0.4;
      case 'chirp': return Math.sin(x*(0.01+x*0.0002) + phase);
      case 'noise': return Math.sin(x*0.05+phase) + (Math.sin(x*1.7+phase*3)*0.15);
      case 'square': return Math.sign(Math.sin(x*0.06+phase));
      case 'step': return Math.tanh(Math.sin(x*0.05+phase)*3);
      case 'sumSine': return (Math.sin(x*0.04+phase)+Math.sin(x*0.09+phase*1.3)+Math.sin(x*0.15+phase*0.6))/3;
      case 'noisyDecay': return Math.sin(x*0.05+phase)*Math.exp(-((x%300)/300));
      default: return Math.sin(x*0.05+phase);
    }
  }
  function frame(){
    const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = 'rgba(255,180,84,.55)';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(255,180,84,.5)'; ctx.shadowBlur = 6;
    ctx.beginPath();
    for(let x=0;x<w;x+=2){
      const y = h/2 + waveAt(x+t, 0) * (h*0.32);
      if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    t += 1.6;
    requestAnimationFrame(frame);
  }
  frame();
}

/* ---- mobile nav toggle ---- */
function initNavToggle(){
  document.getElementById('navToggle').addEventListener('click', ()=>{
    document.getElementById('sidebar').classList.toggle('open');
  });
}

/* ---- boot ---- */
document.addEventListener('DOMContentLoaded', ()=>{
  buildSidebar();
  initScope();
  initNavToggle();
  const hashId = location.hash.replace('#','');
  const startId = (hashId && findTopic(hashId)) ? hashId : CHAPTERS[0].topics[0].id;
  goToTopic(startId);
});
