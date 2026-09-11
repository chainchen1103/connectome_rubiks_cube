"""Self-contained research HTML exports; no server or external assets required.

Edges use zero-based indices into ``graph.neurons``. Large graphs are exported
as explicitly labelled, deterministic subgraphs to keep a browser responsive.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MAX_GRAPH_NODES = 1200
MAX_GRAPH_EDGES = 5000
MAX_SPIKE_FRAMES = 600


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _script_json(value: Any) -> str:
    """Escape HTML parser delimiters, including inside application/json scripts."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _data_label(metadata: Mapping[str, Any]) -> tuple[str, str]:
    provenance = " ".join(
        str(metadata.get(key, ""))
        for key in ("kind", "parent_kind", "source", "source_type", "data_kind", "dataset_kind", "data_source")
    ).lower()
    if metadata.get("kind") == "control":
        return "對照拓撲", "此圖的連線已依對照方法重建，並非原始生物連接；資料來源與對照方法請見中繼資料。"
    if metadata.get("synthetic") is True or "synthetic" in provenance:
        return "合成資料", "合成資料示範；不代表果蠅真實連接或認知能力。"
    if metadata.get("synthetic") is False or any(
        word in provenance.split() for word in ("real", "measured_larval", "measured_larval_subset")
    ) or any(
        word in provenance for word in ("flywire", "hemibrain")
    ):
        return "實際資料匯入", "資料來源與處理方式請見中繼資料；圖形本身不構成生物學驗證。"
    return "資料來源待核對", "請依中繼資料核對來源；圖形本身不構成生物學驗證。"


def _write_html(output_path: str | Path, document: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def _graph_payload(graph: Any, spikes: Any, dt_ms: float) -> dict[str, Any]:
    neurons = list(graph.neurons)
    count = len(neurons)
    if count == 0:
        raise ValueError("Graph must contain at least one neuron")
    dt_ms = float(dt_ms)
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    source = np.asarray(graph.source)
    target = np.asarray(graph.target)
    weights = np.asarray(graph.synapse_count, dtype=float)
    if source.ndim != 1 or target.ndim != 1 or weights.ndim != 1:
        raise ValueError("Graph edges must be one-dimensional arrays")
    if not (len(source) == len(target) == len(weights)):
        raise ValueError("Graph edge arrays must have matching lengths")
    if len(source):
        if source.dtype.kind not in "iu" or target.dtype.kind not in "iu":
            raise ValueError("Edge indices must be integers")
        if min(source.min(), target.min()) < 0 or max(source.max(), target.max()) >= count:
            raise ValueError("Edge index is outside graph.neurons")
    source, target = source.astype(np.int64), target.astype(np.int64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Synapse counts must be finite and nonnegative")
    positions = np.asarray([[n.x, n.y, n.z] for n in neurons], dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("Neuron coordinates must be finite")
    ids = [str(n.id) for n in neurons]
    if len(set(ids)) != count:
        raise ValueError("Neuron IDs must be unique")

    chosen = np.linspace(0, count - 1, min(count, MAX_GRAPH_NODES), dtype=np.int64)
    remap = np.full(count, -1, dtype=np.int64)
    remap[chosen] = np.arange(len(chosen))
    eligible = np.flatnonzero((remap[source] >= 0) & (remap[target] >= 0))
    if len(eligible) > MAX_GRAPH_EDGES:
        # Stable sort makes equal-weight selection deterministic.
        eligible = eligible[np.argsort(-weights[eligible], kind="stable")[:MAX_GRAPH_EDGES]]
    indegree = np.bincount(target, minlength=count)
    outdegree = np.bincount(source, minlength=count)
    nodes = [
        {
            "id": ids[i],
            "type": str(neurons[i].neuron_type),
            "transmitter": str(neurons[i].neurotransmitter),
            "role": str(neurons[i].role),
            "region": str(neurons[i].region),
            "xyz": positions[i].tolist(),
            "original_index": int(i),
            "indegree": int(indegree[i]),
            "outdegree": int(outdegree[i]),
        }
        for i in chosen
    ]
    activity: list[list[int]] = []
    frame_times: list[float] = []
    frame_ends: list[float] = []
    total_frames = 0
    if spikes is not None:
        spike_array = np.asarray(spikes)
        if spike_array.ndim != 2 or spike_array.shape[1] != count:
            raise ValueError("spikes must have shape [time, number_of_neurons]")
        if not np.isin(spike_array, [0, 1]).all():
            raise ValueError("spikes must contain binary values (0 or 1)")
        total_frames = spike_array.shape[0]
        bounds = np.linspace(0, total_frames, min(total_frames, MAX_SPIKE_FRAMES) + 1, dtype=int)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            activity.append(np.flatnonzero(spike_array[start:stop, chosen].any(axis=0)).tolist())
            frame_times.append(float(start * dt_ms))
            frame_ends.append(float(stop * dt_ms))
    metadata = dict(graph.metadata)
    label, caveat = _data_label(metadata)
    return {
        "nodes": nodes,
        "edges": [[int(remap[source[i]]), int(remap[target[i]]), float(weights[i])] for i in eligible],
        "metadata": metadata,
        "label": label,
        "caveat": caveat,
        "original_nodes": count,
        "original_edges": len(source),
        "sampled": count > len(chosen) or len(source) > len(eligible),
        "sampling_method": "依原始節點順序等距抽樣；保留子圖內突觸數較多的邊，同權重時依原始順序。",
        "activity": activity,
        "frame_times_ms": frame_times,
        "frame_ends_ms": frame_ends,
        "original_frames": total_frames,
        "dt_ms": dt_ms,
        "temporal_aggregation": total_frames > MAX_SPIKE_FRAMES,
    }


_STYLE = """
:root{color-scheme:dark;--bg:#0b1320;--panel:#121f30;--line:#294055;--fg:#e7eef7;--muted:#a6b8cc;--accent:#6de0d2;--warm:#ffb86b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Microsoft JhengHei",sans-serif}
header{padding:24px 32px 20px;border-bottom:1px solid var(--line)}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.16em}h1{font-size:27px;line-height:1.3;margin:6px 0 10px;letter-spacing:.02em}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:18px 0 8px}p{margin:8px 0}.muted,small{color:var(--muted)}.badge{display:inline-block;border:1px solid #487366;border-radius:5px;padding:2px 8px;color:var(--accent);font-size:12px}.warning{color:var(--warm)}
button,input,select{font:inherit}button,select,input[type=search]{color:var(--fg);background:#1b3044;border:1px solid #466279;border-radius:6px;padding:7px 10px}button{cursor:pointer}button:hover{background:#29445b}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:3px}input[type=range]{accent-color:var(--accent);vertical-align:middle}input[type=checkbox]{accent-color:var(--accent)}label{display:inline-flex;gap:7px;align-items:center}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 24px;background:var(--panel);border-bottom:1px solid var(--line)}.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;min-height:640px}.stage{position:relative;min-height:640px;background:radial-gradient(ellipse at center,#14253b,#09121e)}canvas{display:block;width:100%;height:100%;min-height:640px;touch-action:none}.overlay{position:absolute;top:16px;left:20px;right:20px;pointer-events:none;font-size:13px}.legend{position:absolute;bottom:20px;left:20px;right:20px;pointer-events:none;display:flex;gap:12px;flex-wrap:wrap;font-size:12px}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.sidebar{background:var(--panel);border-left:1px solid var(--line);padding:22px;overflow-wrap:anywhere}.details{display:grid;grid-template-columns:85px 1fr;gap:6px;font-size:13px}.details dt{color:var(--muted)}.details dd{margin:0}.neighbors{list-style:none;padding:0;margin:0;max-height:180px;overflow:auto}.neighbors li{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:5px 0;font-size:12px}.neighbors button{font-size:12px;padding:3px 6px;max-width:200px;overflow-wrap:anywhere;text-align:left}.search-results{max-height:160px;overflow:auto;display:grid;gap:5px;margin-top:6px}.search-results button{text-align:left;font-size:12px}.footer{border-top:1px solid var(--line);padding:18px 32px;font-size:12px}details{margin-top:14px}summary{cursor:pointer;color:var(--accent)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;color:var(--muted)}.report{max-width:1280px;margin:auto;padding:28px 32px 60px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:26px}.metric,.card{padding:22px;background:var(--panel);border:1px solid var(--line);border-radius:10px}.metric strong{display:block;font-size:28px;line-height:1.4}.metric span{color:var(--muted);font-size:13px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card svg{display:block;width:100%;height:auto}.table-wrap{overflow-x:auto;margin:22px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:500}td.numeric{text-align:right;font-variant-numeric:tabular-nums}.note{padding:15px 18px;border-left:3px solid var(--accent);background:#132737;margin:18px 0;color:var(--muted)}.run-meta{display:flex;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:12px}.run-meta span{background:#203246;padding:2px 7px;border-radius:4px}@media(max-width:850px){.layout{grid-template-columns:1fr}.sidebar{border-left:0;border-top:1px solid var(--line)}.stage,canvas{min-height:480px;height:480px}.cards{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}header,.report{padding-left:20px;padding-right:20px}}@media print{body{background:white;color:black}.card,.metric{break-inside:avoid;background:#f3f6fa;color:black}header{padding:12px}.report{padding:12px}.muted,small,.metric span{color:#445}details{display:block}.cards{grid-template-columns:1fr}svg text{fill:#334155}.note{color:#334155;background:#eef4f8}}
"""


_GRAPH_BODY = r"""
<header><div class="eyebrow">CONNECTOME LAB / STRUCTURE + ACTIVITY</div><h1>神經連接體・三維探索</h1><span id="label" class="badge"></span><p id="caveat" class="muted"></p></header>
<div class="toolbar"><label>腦區 <select id="region"><option value="">全部腦區</option></select></label><label><input id="edges" type="checkbox" checked>顯示方向連線</label><button id="reset">重設視角</button><button id="play">播放活動</button><label>時間 <input id="time" type="range" min="0" max="0" value="0" aria-label="活動時間"></label><output id="clock" aria-live="off">未提供活動資料</output></div>
<main class="layout"><section class="stage"><canvas id="graph" tabindex="0" role="img" aria-label="互動三維神經連接圖。拖曳旋轉、滾輪縮放、點選神經元。鍵盤方向鍵旋轉，加減鍵縮放。"></canvas><div class="overlay"><div id="counts"></div><div id="sampling" class="warning"></div><div class="muted">拖曳旋轉 · 滾輪縮放 · 點選神經元<br>方向鍵旋轉 · ＋／－縮放 · 連線箭頭：來源 → 目標</div></div><div id="legend" class="legend"></div></section><aside class="sidebar"><h2>神經元檢視</h2><label for="search" class="muted">依 ID 或類型搜尋</label><input type="search" id="search" placeholder="輸入神經元 ID…" style="width:100%"><div id="search-results" class="search-results"></div><p id="selection-hint" class="muted">點選圖中的節點，以檢視來源與下游目標。</p><div id="inspector" hidden><dl id="node-details" class="details"></dl><h3 style="color:#72cbff">上游 → 選取神經元</h3><ul id="upstream" class="neighbors"></ul><h3 style="color:#ffb86b">選取神經元 → 下游</h3><ul id="downstream" class="neighbors"></ul><p class="muted"><small>清單依突觸數排序，僅列出所展示子圖內的連線；完整圖入度／出度另列於上方。活動中節點以亮黃色表示。</small></p></div><details><summary>資料與取樣資訊</summary><pre id="metadata"></pre></details></aside></main>
<footer class="footer muted">獨立離線研究產物 · 無外部資源或網路請求 · 座標以原始資料比例投影。連線粗細反映突觸數；三維呈現不代表精確神經生理模擬。</footer>
"""


_GRAPH_SCRIPT = r"""
(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('connectome-data').textContent);
  const $=id=>document.getElementById(id), canvas=$('graph'),ctx=canvas.getContext('2d');
  const palette=['#64d6ca','#88a8ff','#f1a66d','#cb9ef4','#f07b9b','#a8d57a','#75c5eb','#d6c177'];
  const regions=[...new Set(data.nodes.map(n=>n.region))].sort();
  const color=r=>palette[regions.indexOf(r)%palette.length];
  regions.forEach((r,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=r||'（未標註）';$('region').append(o);});
  regions.slice(0,16).forEach(r=>{const item=document.createElement('span'),swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=color(r);item.append(swatch,document.createTextNode(r||'（未標註）'));$('legend').append(item);});
  if(regions.length>16)$('legend').append(document.createTextNode(`另有 ${regions.length-16} 個腦區，可由上方篩選。`));
  $('label').textContent=data.label;$('caveat').textContent=data.caveat;
  $('sampling').textContent=data.sampled?'已取樣：目前為子圖，非完整連接體。':'';
  $('metadata').textContent=JSON.stringify({original_nodes:data.original_nodes,displayed_nodes:data.nodes.length,original_edges:data.original_edges,displayed_edges:data.edges.length,sampled:data.sampled,sampling_method:data.sampling_method,original_frames:data.original_frames,displayed_frames:data.activity.length,temporal_aggregation:data.temporal_aggregation,dt_ms:data.dt_ms,...data.metadata},null,2);
  let yaw=.55,pitch=-.3,zoom=1,selected=null,frame=0,playing=false,lastTick=0,projected=[],drag=null;
  const positions=data.nodes.map(n=>n.xyz),bounds=[0,1,2].map(k=>[Math.min(...positions.map(p=>p[k])),Math.max(...positions.map(p=>p[k]))]);
  const center=bounds.map(b=>(b[0]+b[1])/2),span=Math.max(...bounds.map(b=>b[1]-b[0]),1e-9);
  const xyz=positions.map(p=>p.map((v,k)=>(v-center[k])/span));
  const visible=i=>$('region').value===''||data.nodes[i].region===regions[Number($('region').value)];
  function resize(){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
  function draw(){
    const {width:w,height:h}=canvas.getBoundingClientRect(),scale=Math.min(w,h)*.66*zoom;
    ctx.clearRect(0,0,w,h);
    projected=xyz.map((p,i)=>{const x=p[0]*Math.cos(yaw)-p[2]*Math.sin(yaw),z=p[0]*Math.sin(yaw)+p[2]*Math.cos(yaw),y=p[1]*Math.cos(pitch)-z*Math.sin(pitch),depth=p[1]*Math.sin(pitch)+z*Math.cos(pitch);return {i,x:w/2+x*scale,y:h/2-y*scale,z:depth};});
    let shownEdges=0;
    data.edges.forEach(([s,t,weight])=>{
      if(!visible(s)||!visible(t))return;shownEdges++;if(!$('edges').checked)return;
      const a=projected[s],b=projected[t],incoming=t===selected,outgoing=s===selected,highlight=incoming||outgoing;
      ctx.strokeStyle=incoming?'#72cbff':outgoing?'#ffb86b':'#6688a6';ctx.fillStyle=ctx.strokeStyle;
      ctx.globalAlpha=highlight?.9:(selected===null?.19:.055);ctx.lineWidth=highlight?1.6:Math.min(2,.4+Math.log1p(weight)*.18);
      if(s===t){ctx.beginPath();ctx.arc(a.x+7,a.y-7,9,.4,Math.PI*2);ctx.stroke();return;}
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
      const dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy);if(len<10)return;
      const at=.72,ax=a.x+dx*at,ay=a.y+dy*at,size=highlight?6:3,ux=dx/len,uy=dy/len;
      ctx.beginPath();ctx.moveTo(ax+ux*size,ay+uy*size);ctx.lineTo(ax-ux*size-uy*size*.65,ay-uy*size+ux*size*.65);ctx.lineTo(ax-ux*size+uy*size*.65,ay-uy*size-ux*size*.65);ctx.closePath();ctx.fill();
    });
    const active=new Set(data.activity[frame]||[]),linked=new Set();
    if(selected!==null)data.edges.forEach(([s,t])=>{if(s===selected)linked.add(t);if(t===selected)linked.add(s);});
    projected.slice().sort((a,b)=>a.z-b.z).forEach(p=>{
      if(!visible(p.i))return;const current=p.i===selected,isActive=active.has(p.i),r=current?7:isActive?5:Math.max(2.3,3.5+p.z);
      ctx.globalAlpha=selected===null||current||linked.has(p.i)||isActive?.96:.3;ctx.fillStyle=isActive?'#fff08a':color(data.nodes[p.i].region);
      if(isActive){ctx.shadowBlur=14;ctx.shadowColor='#ffe778';}ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
      if(current){ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,r+4,0,Math.PI*2);ctx.stroke();}
    });ctx.globalAlpha=1;
    $('counts').textContent=`顯示 ${data.nodes.filter((_,i)=>visible(i)).length.toLocaleString()} / ${data.original_nodes.toLocaleString()} 個神經元 · ${shownEdges.toLocaleString()} / ${data.original_edges.toLocaleString()} 條連線`;
  }
  function select(i){selected=i;showInspector();draw();}
  function neighbors(id,edges,upstream){const list=$(id);list.replaceChildren();edges.sort((a,b)=>b[2]-a[2]).forEach(e=>{const i=upstream?e[0]:e[1],li=document.createElement('li'),button=document.createElement('button'),weight=document.createElement('span');button.textContent=data.nodes[i].id;button.addEventListener('click',()=>{if(!visible(i))$('region').value='';select(i);});weight.textContent=`${e[2].toLocaleString()} 突觸`;li.append(button,weight);list.append(li);});if(!edges.length){const li=document.createElement('li');li.textContent='子圖中無連線';list.append(li);}}
  function showInspector(){
    $('selection-hint').hidden=selected!==null;$('inspector').hidden=selected===null;if(selected===null)return;
    const n=data.nodes[selected];$('node-details').replaceChildren();
    [['ID',n.id],['類型',n.type],['傳遞物質',n.transmitter],['角色',n.role],['腦區',n.region],['原始座標',n.xyz.map(v=>Number(v.toPrecision(5))).join(', ')],['完整圖入度',n.indegree],['完整圖出度',n.outdegree]].forEach(([key,value])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=String(value);$('node-details').append(dt,dd);});
    neighbors('upstream',data.edges.filter(e=>e[1]===selected),true);neighbors('downstream',data.edges.filter(e=>e[0]===selected),false);
  }
  function showTime(){const times=data.frame_times_ms;if(!times.length)return;$('time').value=String(frame);$('clock').textContent=data.temporal_aggregation?`${times[frame].toFixed(1)}–${data.frame_ends_ms[frame].toFixed(1)} ms（區間內任一脈衝）`:`${times[frame].toFixed(1)} ms`;}
  function animate(now){if(!playing)return;if(now-lastTick>=100){frame=(frame+1)%data.activity.length;showTime();draw();lastTick=now;}requestAnimationFrame(animate);}
  $('play').disabled=!data.activity.length;$('time').disabled=!data.activity.length;$('time').max=String(Math.max(0,data.activity.length-1));showTime();
  $('play').addEventListener('click',()=>{playing=!playing;$('play').textContent=playing?'暫停活動':'播放活動';if(playing)requestAnimationFrame(animate);});
  $('time').addEventListener('input',()=>{frame=Number($('time').value);showTime();draw();});
  $('region').addEventListener('change',()=>{if(selected!==null&&!visible(selected))select(null);draw();});$('edges').addEventListener('change',draw);
  $('reset').addEventListener('click',()=>{yaw=.55;pitch=-.3;zoom=1;draw();});
  $('search').addEventListener('input',()=>{const query=$('search').value.trim().toLowerCase(),list=$('search-results');list.replaceChildren();if(!query)return;const matches=data.nodes.map((n,i)=>({n,i})).filter(({n})=>n.id.toLowerCase().includes(query)||n.type.toLowerCase().includes(query));matches.slice(0,30).forEach(({n,i})=>{const b=document.createElement('button');b.textContent=`${n.id} · ${n.type}`;b.addEventListener('click',()=>{$('region').value='';select(i);});list.append(b);});const info=document.createElement('small');info.textContent=`展示子圖中 ${matches.length} 個符合項目${matches.length>30?'；列出前 30 個':''}`;list.append(info);});
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag.x)*.007;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag.y)*.007));drag.x=e.clientX;drag.y=e.clientY;draw();});
  canvas.addEventListener('pointerup',e=>{if(!drag)return;const moved=Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY);drag=null;if(moved>5)return;const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,distance=13;projected.forEach(p=>{const d=Math.hypot(p.x-x,p.y-y);if(visible(p.i)&&d<distance){distance=d;best=p.i;}});select(best);});
  canvas.addEventListener('pointercancel',()=>{drag=null;});
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(6,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  canvas.addEventListener('keydown',e=>{let handled=true;switch(e.key){case 'ArrowLeft':yaw-=.1;break;case 'ArrowRight':yaw+=.1;break;case 'ArrowUp':pitch=Math.max(-1.5,pitch-.1);break;case 'ArrowDown':pitch=Math.min(1.5,pitch+.1);break;case '+':case '=':zoom=Math.min(6,zoom*1.1);break;case '-':zoom=Math.max(.25,zoom/1.1);break;default:handled=false;}if(handled){e.preventDefault();draw();}});
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(canvas);else window.addEventListener('resize',resize);resize();
})();
"""


def export_graph_html(graph: Any, output_path: str | Path, spikes: Any = None, dt_ms: float = 1.0) -> Path:
    """Export an offline 3D viewer, optionally with binary ``[time, neuron]`` spikes.

    At most 1,200 neurons and 5,000 edges are embedded. More than 600 spike
    frames are binned with logical OR, and the displayed time ranges disclose
    that aggregation. Neither export operation alters the original arrays.
    """
    payload = _graph_payload(graph, spikes, dt_ms)
    document = (
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>神經連接體・三維探索</title><style>' + _STYLE + "</style></head><body>"
        + _GRAPH_BODY + '<script id="connectome-data" type="application/json">'
        + _script_json(payload) + "</script><script>" + _GRAPH_SCRIPT + "</script></body></html>"
    )
    return _write_html(output_path, document)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _reward_svg(rewards: list[float], title: str) -> str:
    safe_title = escape(title)
    if not rewards:
        return '<p class="muted">本次紀錄未提供逐回合獎勵。</p>'
    width, height, left, right, top, bottom = 600, 230, 52, 22, 22, 42
    low, high = min(rewards), max(rewards)
    if low == high:
        low, high = low - .5, high + .5
    pad = (high - low) * .08
    low, high = low - pad, high + pad
    def point(index: int, value: float) -> tuple[float, float]:
        return left + index / max(1, len(rewards) - 1) * (width - left - right), top + (high - value) / (high - low) * (height - top - bottom)
    # Bound SVG size while retaining extrema in each sequential bucket.
    indices = [0]
    if len(rewards) > 1000:
        reward_array = np.asarray(rewards)
        for chunk in np.array_split(np.arange(len(rewards)), 400):
            indices.extend(sorted({int(chunk[np.argmin(reward_array[chunk])]), int(chunk[np.argmax(reward_array[chunk])])}))
        indices = sorted(set(indices + [len(rewards) - 1]))
    else:
        indices = list(range(len(rewards)))
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(i, rewards[i]) for i in indices))
    grid = []
    for fraction in (0, .5, 1):
        value = low + fraction * (high - low)
        y = point(0, value)[1]
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#294055"/><text x="{left-8}" y="{y+4:.2f}" fill="#a6b8cc" text-anchor="end" font-size="11">{value:.2g}</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{safe_title}：逐回合獎勵曲線">'
        f'<title>{safe_title}：逐回合獎勵曲線，共 {len(rewards)} 回合</title>' + "".join(grid)
        + f'<polyline points="{points}" fill="none" stroke="#6de0d2" stroke-width="2" stroke-linejoin="round"/>'
        + (f'<circle cx="{left}" cy="{point(0,rewards[0])[1]:.2f}" r="3" fill="#6de0d2"/>' if len(rewards) == 1 else "")
        + f'<text x="{left}" y="{height-20}" fill="#a6b8cc" font-size="11">1</text>'
        + f'<text x="{width-right}" y="{height-20}" fill="#a6b8cc" text-anchor="end" font-size="11">{len(rewards)}</text>'
        + f'<text x="{width/2}" y="{height-4}" fill="#a6b8cc" text-anchor="middle" font-size="12">訓練回合</text>'
        + '<text x="7" y="13" fill="#a6b8cc" font-size="11">獎勵</text></svg>'
    )


def _success_svg(before: float, after: float) -> str:
    bars = []
    for i, (label, value, color) in enumerate((("訓練前", before, "#7895b5"), ("訓練後", after, "#6de0d2"))):
        y = 17 + i * 42
        bars.append(f'<text x="0" y="{y+17}" fill="#a6b8cc" font-size="13">{label}</text><rect x="64" y="{y}" width="420" height="24" rx="4" fill="#223448"/><rect x="64" y="{y}" width="{420*value:.2f}" height="24" rx="4" fill="{color}"/><text x="500" y="{y+17}" fill="#e7eef7" font-size="13">{value:.1%}</text>')
    return '<svg viewBox="0 0 600 102" role="img" aria-label="訓練前後成功率"><title>訓練前後成功率</title>' + "".join(bars) + "</svg>"


def export_report_html(report: dict[str, Any], output_path: str | Path) -> Path:
    """Export a static research report with SVG charts and the complete JSON.

    ``runs`` is a list of dictionaries. Success rates are fractions in [0, 1];
    reward curves may contain any finite numbers. Missing optional metrics
    remain visibly absent, rather than being interpreted as zero.
    """
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    runs = report.get("runs", [])
    if not isinstance(runs, list) or any(not isinstance(run, Mapping) for run in runs):
        raise ValueError("report.runs must be a list of mappings")
    metadata = report.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("report.metadata must be a mapping")
    label, caveat = _data_label(metadata)
    cards, rows, improvements = [], [], []
    total_interactions = 0.0
    for index, run in enumerate(runs):
        task, topology, seed = (str(run.get(key, "未提供")) for key in ("task", "topology", "seed"))
        title = f"{task} · {topology} · seed {seed}"
        rewards_raw = run.get("training_rewards", [])
        if not isinstance(rewards_raw, (list, tuple, np.ndarray)) or np.asarray(rewards_raw).ndim != 1:
            raise ValueError("training_rewards must be a one-dimensional sequence")
        rewards = [_finite_number(value, "training_rewards") for value in rewards_raw]
        before, after = run.get("before_success"), run.get("after_success")
        success_chart = '<p class="muted">本次紀錄未提供完整的訓練前後成功率。</p>'
        if before is not None:
            before = _finite_number(before, "before_success")
            if not 0 <= before <= 1:
                raise ValueError("before_success must be in [0, 1]")
        if after is not None:
            after = _finite_number(after, "after_success")
            if not 0 <= after <= 1:
                raise ValueError("after_success must be in [0, 1]")
        if before is not None and after is not None:
            improvements.append(after - before)
            success_chart = _success_svg(before, after)
        interactions = run.get("interactions")
        if interactions is not None:
            interactions = _finite_number(interactions, "interactions")
            if interactions < 0 or not interactions.is_integer():
                raise ValueError("interactions must be a nonnegative integer")
            total_interactions += interactions
        weight_change = run.get("weight_change_l1")
        if weight_change is not None:
            weight_change = _finite_number(weight_change, "weight_change_l1")
            if weight_change < 0:
                raise ValueError("weight_change_l1 must be nonnegative")
        stability = run.get("stability")
        stability_html = "" if stability is None else '<details><summary>穩定性紀錄</summary><pre>' + escape(json.dumps(stability, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)) + "</pre></details>"
        cards.append(
            f'<article class="card"><h2>{escape(task)}</h2><div class="run-meta"><span>{escape(topology)}</span><span>seed {escape(seed)}</span><span>{len(rewards)} 訓練回合</span></div><h3>評估成功率</h3>'
            + success_chart + '<h3>訓練獎勵</h3>' + _reward_svg(rewards, title) + stability_html + "</article>"
        )
        cells = [str(index + 1), escape(task), escape(topology), escape(seed), "—" if before is None else f"{before:.1%}", "—" if after is None else f"{after:.1%}", "—" if before is None or after is None else f"{100*(after-before):+.1f}", "—" if interactions is None else f"{int(interactions):,}", "—" if weight_change is None else f"{weight_change:.5g}"]
        table_cells = [('<td class="numeric">' if j >= 4 else '<td>') + cell + '</td>'
                       for j, cell in enumerate(cells)]
        rows.append("<tr>" + "".join(table_cells) + "</tr>")
    mean_change = "—" if not improvements else f"{100*np.mean(improvements):+.1f}"
    interaction_summary = f"{int(total_interactions):,}" if any(run.get("interactions") is not None for run in runs) else "—"
    document = (
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>神經連接體・學習實驗報告</title><style>'
        + _STYLE + '</style></head><body><header><div class="eyebrow">CONNECTOME LAB / EXPERIMENT REPORT</div><h1>神經連接體・學習實驗報告</h1><span class="badge">'
        + escape(label) + '</span><p class="muted">' + escape(caveat) + '</p></header><main class="report"><div class="metrics">'
        + f'<div class="metric"><strong>{len(runs)}</strong><span>實驗紀錄</span></div><div class="metric"><strong>{len({str(run.get("task", "")) for run in runs})}</strong><span>任務種類</span></div><div class="metric"><strong>{interaction_summary}</strong><span>已記錄環境互動總數</span></div><div class="metric"><strong>{mean_change}</strong><span>平均成功率變化（百分點）</span></div></div>'
        + '<p class="note">本報告呈現各次實驗的觀測結果。跨任務平均值僅供描述；請以相同任務、相同評估條件與多個隨機種子比較拓撲效果。單次改善不構成生物學合理性或統計顯著性的證據。</p>'
        + '<h2>實驗比較</h2><div class="table-wrap"><table><thead><tr><th>#</th><th>任務</th><th>拓撲</th><th>種子</th><th>訓練前</th><th>訓練後</th><th>變化（百分點）</th><th>互動數</th><th>權重變化 L1</th></tr></thead><tbody>'
        + ("".join(rows) or '<tr><td colspan="9">尚無實驗紀錄。</td></tr>') + '</tbody></table></div><section class="cards">' + "".join(cards)
        + '</section><details><summary>完整實驗資料（JSON）</summary><pre>' + escape(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False))
        + '</pre></details></main><footer class="footer muted">獨立離線研究產物 · 所有圖表與數據內嵌於本檔案 · 可使用瀏覽器列印或另存 PDF</footer><script id="report-data" type="application/json">'
        + _script_json(report) + '</script></body></html>'
    )
    return _write_html(output_path, document)
