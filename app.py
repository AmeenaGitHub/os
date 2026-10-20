"""
SDG-Based Round Robin Scheduling Visualizer  (OS Assignment 1)
================================================================
Covers BOTH practical activities:
  * Activity 1 - Round Robin WITHOUT context-switch overhead  (quantum 2 ms vs 3 ms)
  * Activity 2 - Round Robin WITH    context-switch overhead  (quantum 2 ms vs 4 ms)

Run it:
    pip install streamlit pandas
    streamlit run app.py

Features
  * You choose the number of processes and type each process's Arrival Time / Burst Time
  * Interactive step-by-step visualizer (play / pause / step / scrub / speed):
      not-yet-arrived -> ready queue -> CPU -> finished, live Gantt chart, live process
      table and a running event log
  * Gantt chart with execution intervals AND every context-switch interval shown separately
  * CT, TAT, WT, RT of every process; average TAT / WT / RT; context-switch count
  * Activity 2 extras: total context-switch overhead and CPU efficiency
  * Tool-generated comparison tables, charts and a quantum recommendation
"""

import json
from collections import deque

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PALETTE = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
    "#14b8a6", "#f97316", "#84cc16", "#06b6d4", "#a855f7", "#eab308",
]

ACT1_DEFAULT = [
    ("Water-quality monitoring", 0, 5),
    ("Waste-collection scheduling", 1, 3),
    ("Traffic-pollution analysis", 2, 4),
    ("Public energy-use monitoring", 3, 2),
]
ACT2_DEFAULT = [
    ("Flood-warning analysis", 0, 5),
    ("Air-quality alert generation", 1, 4),
    ("Emergency evacuation route update", 2, 3),
    ("Renewable-grid status monitoring", 3, 2),
]


# ════════════════════════════════════════════════════════════════════════════
#  1.  ROUND ROBIN SIMULATION ENGINE
# ════════════════════════════════════════════════════════════════════════════
def simulate_rr(procs, quantum, cs_time=0):
    """
    Millisecond-by-millisecond Round Robin simulation.

    procs   : list of dicts  {pid, task, at, bt}
    quantum : time quantum in ms
    cs_time : context-switch overhead in ms (0 = ignore overhead)

    Rules used
      * A process arriving at the same instant a slice ends joins the ready queue
        BEFORE the pre-empted process (standard textbook convention).
      * Overhead is charged only when the CPU changes from one process to a
        DIFFERENT process - never before the first process, never after the last,
        and not when the same process simply continues or after an idle gap.
    """
    pid_list = [p["pid"] for p in procs]
    at = {p["pid"]: p["at"] for p in procs}
    bt = {p["pid"]: p["bt"] for p in procs}
    rem = dict(bt)

    incoming = deque(sorted(pid_list, key=lambda x: (at[x], pid_list.index(x))))
    ready = deque()
    ct, first = {}, {}
    ticks = []

    t = 0
    cur = pending = last = None
    used = cs_left = 0
    switches = 0
    slice_no = 0
    done = 0
    prev_mode = None
    log = []

    def begin(pid):
        """Put pid on the CPU and start a fresh time slice."""
        nonlocal cur, used, slice_no
        cur, used = pid, 0
        slice_no += 1
        if pid not in first:
            first[pid] = t
            log.append(f"{pid} gets the CPU for the first time (response time = {t - at[pid]} ms)")
        else:
            log.append(f"{pid} resumes on the CPU")

    while True:
        log = []

        # 1) arrivals at this instant join the queue first
        while incoming and at[incoming[0]] <= t:
            pid = incoming.popleft()
            ready.append(pid)
            log.append(f"{pid} arrives and joins the ready queue")

        # 2) did the running slice just end?
        if cur is not None and (rem[cur] == 0 or used == quantum):
            if rem[cur] == 0:
                ct[cur] = t
                done += 1
                log.append(f"{cur} finishes  (completion time = {t} ms)")
            else:
                ready.append(cur)
                log.append(f"{cur} used its full quantum ({rem[cur]} ms left) -> back of the ready queue")
            last, cur = cur, None

        # 3) context switch finished -> start the process we switched to
        if pending is not None and cs_left == 0:
            nxt, pending = pending, None
            begin(nxt)

        # 4) CPU free -> dispatch next process (or finish / idle)
        if cur is None and pending is None:
            if done == len(procs):
                log.append(f"All processes finished at t = {t} ms")
                ticks.append(dict(t=t, mode="end", pid=None, frm=None, queue=[], rem=dict(rem),
                                  used=0, csLeft=0, key="end", log=log))
                break
            if ready:
                nxt = ready.popleft()
                if last is not None and last != nxt:
                    switches += 1
                    if cs_time > 0:
                        pending, cs_left = nxt, cs_time
                        log.append(f"Context switch {last} -> {nxt} begins ({cs_time} ms overhead)")
                    else:
                        log.append(f"Context switch {last} -> {nxt} (overhead ignored)")
                        begin(nxt)
                else:
                    if last == nxt:
                        log.append(f"{nxt} is the only ready process -> continues, no context switch")
                    begin(nxt)
            else:
                last = None
                if prev_mode != "idle":
                    log.append("CPU idle - no process is ready")

        # 5) record what happens during [t, t+1)
        if cur is not None:
            mode, shown, key = "exec", cur, f"x{slice_no}"
        elif pending is not None:
            mode, shown, key = "cs", pending, f"c{switches}"
        else:
            mode, shown, key = "idle", None, "idle"
        ticks.append(dict(t=t, mode=mode, pid=shown, frm=last if mode == "cs" else None,
                          queue=list(ready), rem=dict(rem), used=used, csLeft=cs_left,
                          key=key, log=log))
        prev_mode = mode

        # 6) advance one millisecond
        if cur is not None:
            rem[cur] -= 1
            used += 1
        elif pending is not None:
            cs_left -= 1
        t += 1
        if t > 200000:                      # safety net
            raise RuntimeError("Simulation did not terminate")

    # ── merge ticks into Gantt segments ─────────────────────────────────────
    segs = []
    for tk in ticks[:-1]:
        if segs and segs[-1]["key"] == tk["key"] and segs[-1]["end"] == tk["t"]:
            segs[-1]["end"] += 1
        else:
            segs.append(dict(type=tk["mode"], pid=tk["pid"], start=tk["t"], end=tk["t"] + 1,
                             key=tk["key"], frm=tk["frm"]))

    # ── per-process metrics ─────────────────────────────────────────────────
    rows = []
    for i, p in enumerate(procs):
        pid = p["pid"]
        tat = ct[pid] - at[pid]
        rows.append(dict(pid=pid, task=p["task"], at=at[pid], bt=bt[pid], ct=ct[pid],
                         tat=tat, wt=tat - bt[pid], rt=first[pid] - at[pid],
                         first=first[pid], color=PALETTE[i % len(PALETTE)]))

    n = len(rows)
    total_burst = sum(bt.values())
    overhead = switches * cs_time
    idle = sum(s["end"] - s["start"] for s in segs if s["type"] == "idle")
    metrics = dict(
        avg_tat=sum(r["tat"] for r in rows) / n,
        avg_wt=sum(r["wt"] for r in rows) / n,
        avg_rt=sum(r["rt"] for r in rows) / n,
        max_rt=max(r["rt"] for r in rows),
        switches=switches,
        overhead=overhead,
        total_burst=total_burst,
        T=t,
        idle=idle,
        efficiency=100.0 * total_burst / (total_burst + overhead),
    )
    return dict(q=quantum, cs=cs_time, rows=rows, segs=segs, ticks=ticks, T=t, m=metrics)


# ════════════════════════════════════════════════════════════════════════════
#  2.  INTERACTIVE VISUALIZER (self-contained HTML + JS, drawn from simulation data)
# ════════════════════════════════════════════════════════════════════════════
VIZ_HTML = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
:root{--dur:200ms}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.vz{background:#0b1220;color:#e5e7eb;border-radius:14px;padding:16px;border:1px solid #1f2a44}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between}
.title{font-weight:700;font-size:15px}
.title small{color:#94a3b8;font-weight:500;margin-left:8px}
.clock{font-variant-numeric:tabular-nums;font-size:20px;font-weight:800;background:#111b30;border:1px solid #23314f;border-radius:10px;padding:3px 14px}
.clock span{color:#38bdf8}
.ctrls{margin-top:10px;justify-content:flex-start}
button,select{background:#16233f;color:#e5e7eb;border:1px solid #2a3b61;border-radius:8px;padding:7px 11px;font-size:13px;cursor:pointer;font-weight:600;font-family:inherit}
button:hover{background:#1e3057}
button.primary{background:#2563eb;border-color:#3b82f6;min-width:92px}
button.primary:hover{background:#1d4ed8}
.sp{flex:1}
input[type=range]{width:100%;margin:10px 0 0;accent-color:#38bdf8}
.flow{display:flex;gap:8px;align-items:stretch;margin-top:12px;flex-wrap:wrap}
.panel{background:#0f1a2f;border:1px solid #1f2d4d;border-radius:12px;padding:10px 12px;min-height:150px}
.panel h4{margin:0 0 8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;font-weight:700}
.panel h4 small{text-transform:none;letter-spacing:0;font-weight:500;color:#64748b}
.pnew{flex:0 0 150px}.pready{flex:1 1 230px}.pcpu{flex:0 0 230px;border-color:#2b4a85;box-shadow:0 0 0 1px #1d3a6e inset}.pfin{flex:0 0 170px}
.arrow{align-self:center;color:#475a82;font-size:20px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{min-width:52px;padding:6px 8px;border-radius:9px;color:#fff;text-align:center;font-size:12px;line-height:1.25;box-shadow:0 2px 6px rgba(0,0,0,.35)}
.chip b{display:block;font-size:14px}
.chip i{font-style:normal;opacity:.92;font-size:11px}
.chip.front{outline:2px dashed #fbbf24;outline-offset:2px}
.empty{color:#51607f;font-size:12px}
.cpuchip{border-radius:12px;padding:10px;color:#fff;text-align:center;box-shadow:0 4px 14px rgba(0,0,0,.4);transition:background-color var(--dur) linear}
.cpuchip b{font-size:22px;display:block}
.cpuchip span{font-size:11px;opacity:.92}
.cpuchip.cs{background:repeating-linear-gradient(45deg,#64748b,#64748b 8px,#475569 8px,#475569 16px)}
.cpuchip.idle{background:#1e293b;color:#94a3b8}
.prog{height:8px;background:#1e2b4a;border-radius:6px;overflow:hidden;margin-top:10px}
.prog>div{height:100%;width:0;background:#38bdf8;border-radius:6px;transition:width var(--dur) linear}
.meta{font-size:11.5px;color:#94a3b8;margin-top:5px}
.meta b{color:#e5e7eb}
.gpanel{min-height:0;margin-top:12px}
.legend{display:inline-flex;gap:10px;flex-wrap:wrap;margin-left:10px;text-transform:none;letter-spacing:0;font-weight:500}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.gwrap{overflow-x:auto;padding:6px 0 4px}
.gbox{position:relative;margin-left:12px}
.gantt{position:relative;height:52px}
.seg{position:absolute;top:0;height:52px;overflow:hidden;white-space:nowrap;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff;border-radius:6px;box-shadow:inset -2px 0 0 #0b1220;transition:width var(--dur) linear}
.seg.cs{background:repeating-linear-gradient(45deg,#64748b,#64748b 5px,#475569 5px,#475569 10px);font-size:10px}
.seg.idle{background:#1e293b;color:#64748b;font-weight:500;font-size:11px}
.axis{position:relative;height:26px;margin-top:2px}
.tick{position:absolute;top:0;width:1px;height:6px;background:#52628a}
.tl{position:absolute;top:8px;transform:translateX(-50%);font-size:11px;color:#94a3b8;font-variant-numeric:tabular-nums}
.cursor{position:absolute;top:-4px;bottom:0;width:2px;background:#f43f5e;transition:left var(--dur) linear;z-index:5;border-radius:2px}
.tbl{min-height:0;margin-top:12px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{text-align:left;color:#94a3b8;font-weight:600;padding:6px 8px;border-bottom:1px solid #22314f;font-size:11px;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid #16233f;white-space:nowrap}
tfoot td{font-weight:700;color:#e5e7eb;border-bottom:none}
tr.run td{background:#10264a}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.badge{padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600}
.b-new{background:#1e293b;color:#94a3b8}.b-ready{background:#3b2f06;color:#fbbf24}.b-run{background:#06324a;color:#38bdf8}.b-cs{background:#2a2f3c;color:#cbd5e1}.b-done{background:#052e1b;color:#34d399}
.rbar{display:inline-block;width:60px;height:6px;background:#1e2b4a;border-radius:4px;overflow:hidden;vertical-align:middle;margin-left:6px}
.rbar i{display:block;height:100%;width:0;transition:width var(--dur) linear}
.logp{min-height:0;margin-top:12px}
.log{height:150px;overflow-y:auto;font-size:12.5px}
.log div{padding:2px 0;color:#cbd5e1}
.log .tt{display:inline-block;min-width:58px;color:#38bdf8;font-weight:700}
.ganttonly .hide-g{display:none!important}
.ganttonly .gpanel{margin-top:8px}
</style></head>
<body>
<div class="vz" id="vz">
  <div class="row">
    <div class="title" id="title"></div>
    <div class="clock hide-g">t = <span id="tnow">0</span> ms</div>
  </div>

  <div class="row ctrls hide-g">
    <button id="bReset" title="Back to t = 0">&#9198; Reset</button>
    <button id="bBack" title="Back 1 ms">&#9664; 1 ms</button>
    <button id="play" class="primary">&#9654; Play</button>
    <button id="bStep" title="Forward 1 ms">1 ms &#9654;</button>
    <button id="bPrevEv" title="Previous scheduling event">&#8676; Prev event</button>
    <button id="bNextEv" title="Next scheduling event">Next event &#8677;</button>
    <button id="bEnd" title="Jump to the end">&#9197; End</button>
    <span class="sp"></span>
    <label style="font-size:12px;color:#94a3b8">Speed
      <select id="speed">
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option>
        <option value="4">4x</option><option value="8">8x</option>
      </select></label>
  </div>
  <input type="range" id="scrub" class="hide-g" min="0" max="0" value="0" step="1">

  <div class="flow hide-g">
    <div class="panel pnew"><h4>Not yet arrived</h4><div id="newq" class="chips"></div></div>
    <div class="arrow">&#10140;</div>
    <div class="panel pready"><h4>Ready queue <small>(front &rarr; back)</small></h4><div id="rq" class="chips"></div></div>
    <div class="arrow">&#10140;</div>
    <div class="panel pcpu"><h4>CPU</h4>
      <div id="cpuchip" class="cpuchip idle"><b id="cpuname">IDLE</b><span id="cpusub">&nbsp;</span></div>
      <div class="prog"><div id="cpubar"></div></div>
      <div class="meta" id="cpum1">&nbsp;</div><div class="meta" id="cpum2">&nbsp;</div>
    </div>
    <div class="arrow">&#10140;</div>
    <div class="panel pfin"><h4>Finished</h4><div id="fin" class="chips"></div></div>
  </div>

  <div class="panel gpanel">
    <h4>Gantt chart <span class="legend" id="legend"></span></h4>
    <div class="gwrap" id="gwrap"><div class="gbox" id="gbox">
      <div class="gantt" id="gantt"></div><div class="axis" id="axis"></div><div class="cursor" id="cursor"></div>
    </div></div>
  </div>

  <div class="panel tbl hide-g">
    <h4>Process table <small>(fills in live as the simulation runs)</small></h4>
    <table><thead><tr>
      <th>Process</th><th>Task</th><th>AT</th><th>BT</th><th>Remaining</th><th>State</th><th>RT</th><th>CT</th><th>TAT</th><th>WT</th>
    </tr></thead><tbody id="tbody"></tbody>
    <tfoot><tr><td colspan="6">Average</td><td id="aRT">&mdash;</td><td></td><td id="aTAT">&mdash;</td><td id="aWT">&mdash;</td></tr></tfoot></table>
  </div>

  <div class="panel logp hide-g"><h4>Event log <small>(newest first)</small></h4><div class="log" id="log"></div></div>
</div>

<script>
(function(){
const D = __DATA__;
const P = D.procs, S = D.segs, K = D.ticks, T = D.T, Q = D.quantum, CSL = D.cs;
const $ = id => document.getElementById(id);
const col = {}, byPid = {};
P.forEach(p => { col[p.pid] = p.color; byPid[p.pid] = p; });
let cur = 0, playing = false, timer = null, speed = 1, px = 40;
const BASE = 700;

$('title').innerHTML = '';
const tt = document.createElement('span'); tt.textContent = D.title; $('title').appendChild(tt);
const sm = document.createElement('small');
sm.textContent = 'Round Robin · quantum = ' + Q + ' ms · context-switch overhead = ' + CSL + ' ms';
$('title').appendChild(sm);
$('scrub').max = T;
if (D.ganttOnly) document.getElementById('vz').classList.add('ganttonly');

// legend
$('legend').innerHTML = P.map(p => '<span><i style="background:'+p.color+'"></i>'+p.pid+'</span>').join('') +
  (CSL > 0 ? '<span><i style="background:repeating-linear-gradient(45deg,#64748b,#64748b 3px,#475569 3px,#475569 6px)"></i>Context switch</span>' : '') +
  (S.some(s => s.type === 'idle') ? '<span><i style="background:#1e293b;border:1px solid #475569"></i>Idle</span>' : '');

// process table (built once, updated every tick)
const rows = {};
P.forEach(p => {
  const tr = document.createElement('tr');
  tr.innerHTML = '<td><span class="dot" style="background:'+p.color+'"></span><b>'+p.pid+'</b></td>' +
    '<td></td><td>'+p.at+'</td><td>'+p.bt+'</td>' +
    '<td><span class="rem">'+p.bt+'</span><span class="rbar"><i style="background:'+p.color+'"></i></span></td>' +
    '<td><span class="badge b-new">Not arrived</span></td><td class="rt">&mdash;</td><td class="ct">&mdash;</td><td class="tat">&mdash;</td><td class="wt">&mdash;</td>';
  tr.children[1].textContent = p.task;
  $('tbody').appendChild(tr);
  rows[p.pid] = { tr, rem: tr.querySelector('.rem'), bar: tr.querySelector('.rbar i'), st: tr.querySelector('.badge'),
                  rt: tr.querySelector('.rt'), ct: tr.querySelector('.ct'), tat: tr.querySelector('.tat'), wt: tr.querySelector('.wt') };
});
const avg = k => (P.reduce((a, p) => a + p[k], 0) / P.length).toFixed(2);

// gantt (built once per width)
let segEls = [];
function buildGantt(){
  const W = $('gwrap').clientWidth; if (!W) return;
  px = Math.max(26, Math.min(64, Math.floor((W - 24) / Math.max(T, 1))));
  const g = $('gantt'); g.innerHTML = ''; segEls = [];
  g.style.width = (T * px) + 'px'; $('gbox').style.width = (T * px + 14) + 'px';
  S.forEach(s => {
    const d = document.createElement('div');
    d.className = 'seg ' + s.type;
    d.style.left = (s.start * px) + 'px'; d.style.width = '0px';
    if (s.type === 'exec') { d.style.background = col[s.pid]; d.textContent = s.pid; }
    else if (s.type === 'cs') { d.textContent = 'CS'; d.title = 'Context switch ' + s.frm + ' → ' + s.pid; }
    else { d.textContent = 'idle'; }
    g.appendChild(d); segEls.push(d);
  });
  const ax = $('axis'); ax.innerHTML = '';
  let step = 1; for (const st of [1,2,5,10,20,50,100,200,500,1000]) { step = st; if (st * px >= 28) break; }
  for (let m = 0; m <= T; m++) {
    const k = document.createElement('div'); k.className = 'tick'; k.style.left = (m * px) + 'px'; ax.appendChild(k);
    if (m % step === 0) { const l = document.createElement('div'); l.className = 'tl'; l.style.left = (m * px) + 'px'; l.textContent = m; ax.appendChild(l); }
  }
  render();
}

function chip(pid, sub, cls){
  return '<div class="chip '+(cls||'')+'" style="background:'+col[pid]+'"><b>'+pid+'</b><i>'+sub+'</i></div>';
}

function render(){
  const k = K[cur];
  document.documentElement.style.setProperty('--dur', (D.ganttOnly ? 0 : (playing ? BASE / speed : 120)) + 'ms');
  $('tnow').textContent = cur; $('scrub').value = cur;

  // queues
  const nw = P.filter(p => p.at > cur);
  $('newq').innerHTML = nw.length ? nw.map(p => chip(p.pid, 'arrives @' + p.at)).join('') : '<span class="empty">—</span>';
  $('rq').innerHTML = k.queue.length ? k.queue.map((pid, i) => chip(pid, 'rem ' + k.rem[pid], i === 0 ? 'front' : '')).join('') : '<span class="empty">empty</span>';
  const fn = P.filter(p => p.ct <= cur);
  $('fin').innerHTML = fn.length ? fn.map(p => chip(p.pid, 'CT ' + p.ct)).join('') : '<span class="empty">—</span>';

  // CPU
  const ce = $('cpuchip');
  if (k.mode === 'exec') {
    ce.className = 'cpuchip'; ce.style.background = col[k.pid];
    $('cpuname').textContent = k.pid; $('cpusub').textContent = byPid[k.pid].task;
    $('cpubar').style.width = Math.min(100, k.used / Q * 100) + '%';
    $('cpum1').innerHTML = 'Remaining burst: <b>' + k.rem[k.pid] + ' ms</b>';
    $('cpum2').innerHTML = 'Quantum used: <b>' + k.used + ' / ' + Q + ' ms</b>';
  } else if (k.mode === 'cs') {
    ce.className = 'cpuchip cs'; ce.style.background = '';
    $('cpuname').textContent = '⚙ CONTEXT SWITCH'; $('cpuname').style.fontSize = '15px';
    $('cpusub').textContent = k.frm + ' → ' + k.pid;
    $('cpubar').style.width = ((CSL - k.csLeft) / CSL * 100) + '%';
    $('cpum1').innerHTML = 'Overhead: <b>' + (CSL - k.csLeft) + ' / ' + CSL + ' ms</b>';
    $('cpum2').innerHTML = 'Next to run: <b>' + k.pid + '</b>';
  } else if (k.mode === 'idle') {
    ce.className = 'cpuchip idle'; ce.style.background = '';
    $('cpuname').textContent = 'IDLE'; $('cpusub').textContent = 'no process ready';
    $('cpubar').style.width = '0%'; $('cpum1').innerHTML = '&nbsp;'; $('cpum2').innerHTML = '&nbsp;';
  } else {
    ce.className = 'cpuchip idle'; ce.style.background = '';
    $('cpuname').textContent = '✔ DONE'; $('cpusub').textContent = 'all processes finished';
    $('cpubar').style.width = '100%'; $('cpum1').innerHTML = '&nbsp;'; $('cpum2').innerHTML = '&nbsp;';
  }
  if (k.mode !== 'cs') $('cpuname').style.fontSize = '';

  // gantt
  segEls.forEach((el, i) => {
    const s = S[i];
    el.style.width = (Math.max(0, Math.min(cur, s.end) - s.start) * px) + 'px';
  });
  $('cursor').style.left = (cur * px) + 'px';
  $('cursor').style.display = D.ganttOnly ? 'none' : 'block';
  if (!D.ganttOnly) {
    const gw = $('gwrap'), x = cur * px;
    if (x > gw.scrollLeft + gw.clientWidth - 40) gw.scrollLeft = x - gw.clientWidth + 120;
    if (x < gw.scrollLeft) gw.scrollLeft = Math.max(0, x - 40);
  }

  // table
  const label = { 'new':'Not arrived', ready:'Ready', run:'Running', cs:'Switching in', done:'Done' };
  P.forEach(p => {
    const r = rows[p.pid], rem = k.rem[p.pid];
    r.rem.textContent = rem; r.bar.style.width = ((p.bt - rem) / p.bt * 100) + '%';
    let st = 'new';
    if (p.ct <= cur) st = 'done';
    else if (k.mode === 'exec' && k.pid === p.pid) st = 'run';
    else if (k.mode === 'cs' && k.pid === p.pid) st = 'cs';
    else if (p.at <= cur) st = 'ready';
    r.st.className = 'badge b-' + st; r.st.textContent = label[st];
    r.tr.className = st === 'run' ? 'run' : '';
    r.rt.textContent = p.first <= cur ? p.rt : '—';
    r.ct.textContent = p.ct <= cur ? p.ct : '—';
    r.tat.textContent = p.ct <= cur ? p.tat : '—';
    r.wt.textContent = p.ct <= cur ? p.wt : '—';
  });
  const fin = cur >= T;
  $('aRT').textContent = fin ? avg('rt') : '—'; $('aTAT').textContent = fin ? avg('tat') : '—'; $('aWT').textContent = fin ? avg('wt') : '—';

  // log (newest first)
  let html = '', n = 0;
  for (let i = cur; i >= 0 && n < 120; i--) {
    for (let j = K[i].log.length - 1; j >= 0 && n < 120; j--, n++) {
      const d = document.createElement('div'); const a = document.createElement('span'); a.className = 'tt'; a.textContent = 't = ' + i;
      d.appendChild(a); d.appendChild(document.createTextNode(K[i].log[j])); html += d.outerHTML;
    }
  }
  $('log').innerHTML = html || '<div style="color:#51607f">Press Play or step forward to start.</div>';
}

// controls
function pause(){ playing = false; clearInterval(timer); $('play').innerHTML = '&#9654; Play'; render(); }
function play(){
  if (cur >= T) cur = 0;
  playing = true; $('play').innerHTML = '&#9208; Pause';
  clearInterval(timer);
  timer = setInterval(() => { if (cur >= T) { pause(); return; } cur++; render(); if (cur >= T) pause(); }, BASE / speed);
  render();
}
$('play').onclick = () => playing ? pause() : play();
$('bReset').onclick = () => { pause(); cur = 0; render(); };
$('bEnd').onclick = () => { pause(); cur = T; render(); };
$('bBack').onclick = () => { pause(); cur = Math.max(0, cur - 1); render(); };
$('bStep').onclick = () => { pause(); cur = Math.min(T, cur + 1); render(); };
$('bNextEv').onclick = () => { pause(); let i = cur + 1; while (i < T && !K[i].log.length) i++; cur = Math.min(T, i); render(); };
$('bPrevEv').onclick = () => { pause(); let i = cur - 1; while (i > 0 && !K[i].log.length) i--; cur = Math.max(0, i); render(); };
$('scrub').oninput = e => { cur = +e.target.value; render(); };
$('speed').onchange = e => { speed = +e.target.value; if (playing) play(); };

if (D.ganttOnly) cur = T;
buildGantt();
if (window.ResizeObserver) { let last = $('gwrap').clientWidth; new ResizeObserver(() => { const w = $('gwrap').clientWidth; if (w && w !== last) { last = w; buildGantt(); } }).observe($('gwrap')); }
window.addEventListener('load', buildGantt);
render();
})();
</script>
</body></html>
'''


def build_html(res, title, gantt_only=False):
    data = dict(title=title, procs=res["rows"], segs=res["segs"], ticks=res["ticks"],
                T=res["T"], quantum=res["q"], cs=res["cs"], ganttOnly=gantt_only)
    payload = json.dumps(data).replace("<", "\\u003c")
    return VIZ_HTML.replace("__DATA__", payload)


def text_gantt(res):
    """Plain-text Gantt chart that can be pasted into a report."""
    out = []
    for s in res["segs"]:
        lab = s["pid"] if s["type"] == "exec" else ("CS" if s["type"] == "cs" else "IDLE")
        out.append(f"{s['start']} |{lab}| ")
    return "".join(out) + str(res["T"])


# ════════════════════════════════════════════════════════════════════════════
#  3.  STREAMLIT UI HELPERS
# ════════════════════════════════════════════════════════════════════════════
def fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def show_df(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except Exception:                                    # older Streamlit versions
        st.dataframe(df, hide_index=True, use_container_width=True)


def read_processes(key, defaults):
    n = st.number_input("Number of processes", min_value=1, max_value=12,
                        value=len(defaults), step=1, key=f"{key}_n")
    h = st.columns([1, 4, 2, 2])
    h[0].markdown("**Process**")
    h[1].markdown("**Task (optional label)**")
    h[2].markdown("**Arrival time (ms)**")
    h[3].markdown("**Burst time (ms)**")
    procs = []
    for i in range(int(n)):
        d = defaults[i] if i < len(defaults) else (f"Task {i + 1}", i, 3)
        c = st.columns([1, 4, 2, 2])
        c[0].markdown(f"<div style='padding-top:8px'><b>P{i + 1}</b></div>", unsafe_allow_html=True)
        task = c[1].text_input("task", d[0], key=f"{key}_task{i}", label_visibility="collapsed")
        a = c[2].number_input("AT", 0, 1000, d[1], 1, key=f"{key}_at{i}", label_visibility="collapsed")
        b = c[3].number_input("BT", 1, 1000, d[2], 1, key=f"{key}_bt{i}", label_visibility="collapsed")
        procs.append(dict(pid=f"P{i + 1}", task=task.strip() or f"Task {i + 1}", at=int(a), bt=int(b)))
    return procs


def results_df(res):
    return pd.DataFrame([{"Process": r["pid"], "Task": r["task"], "AT": r["at"], "BT": r["bt"],
                          "CT": r["ct"], "TAT": r["tat"], "WT": r["wt"], "RT": r["rt"]}
                         for r in res["rows"]])


def show_run(res, with_overhead):
    m = res["m"]
    label = f"Quantum = {res['q']} ms"

    st.markdown("##### ▶ Interactive visualizer")
    st.caption("Press **Play**, or step one millisecond / one scheduling event at a time. "
               "Drag the slider to scrub through time.")
    components.html(build_html(res, label), height=860 + 38 * len(res["rows"]), scrolling=True)

    st.markdown("##### 📊 Complete Gantt chart")
    components.html(build_html(res, label, gantt_only=True), height=215, scrolling=False)
    st.code(text_gantt(res), language=None)

    st.markdown("##### 🧮 Results for every process")
    show_df(results_df(res))

    st.markdown("##### 📌 Averages and totals")
    if with_overhead:
        c = st.columns(4)
        c[0].metric("Average turnaround time", f"{m['avg_tat']:.2f} ms")
        c[1].metric("Average waiting time", f"{m['avg_wt']:.2f} ms")
        c[2].metric("Average response time", f"{m['avg_rt']:.2f} ms")
        c[3].metric("Total time (makespan)", f"{m['T']} ms")
        c = st.columns(3)
        c[0].metric("Context switches", m["switches"])
        c[1].metric("Total context-switch overhead", f"{m['overhead']} ms")
        c[2].metric("CPU efficiency", f"{m['efficiency']:.2f} %",
                    help="useful burst time ÷ (useful burst time + context-switch overhead)")
        if m["idle"]:
            st.caption(f"CPU was idle for {m['idle']} ms (no process had arrived yet).")
    else:
        c = st.columns(4)
        c[0].metric("Average turnaround time", f"{m['avg_tat']:.2f} ms")
        c[1].metric("Average waiting time", f"{m['avg_wt']:.2f} ms")
        c[2].metric("Average response time", f"{m['avg_rt']:.2f} ms")
        c[3].metric("Context switches", m["switches"])


def show_comparison(a, b, with_overhead, key):
    qa, qb = a["q"], b["q"]
    ma, mb = a["m"], b["m"]

    metrics = [
        ("Average turnaround time (ms)", "avg_tat", True),
        ("Average waiting time (ms)", "avg_wt", True),
        ("Average response time (ms)", "avg_rt", True),
        ("Maximum response time (ms)", "max_rt", True),
        ("Context switches", "switches", True),
        ("Total completion time (ms)", "T", True),
    ]
    if with_overhead:
        metrics += [("Total context-switch overhead (ms)", "overhead", True),
                    ("CPU efficiency (%)", "efficiency", False)]

    rows = []
    for name, k, lower_better in metrics:
        va, vb = round(ma[k], 6), round(mb[k], 6)
        if va == vb:
            better = "Tie"
        elif (va < vb) == lower_better:
            better = f"Q = {qa} ms"
        else:
            better = f"Q = {qb} ms"
        rows.append({"Metric": name, f"Q = {qa} ms": fmt(ma[k]), f"Q = {qb} ms": fmt(mb[k]), "Better": better})
    st.markdown("##### 📋 Tool-generated comparison table")
    show_df(pd.DataFrame(rows))

    st.markdown("##### 👥 Per-process comparison")
    per = []
    for ra, rb in zip(a["rows"], b["rows"]):
        per.append({"Process": ra["pid"],
                    f"CT @Q={qa}": ra["ct"], f"CT @Q={qb}": rb["ct"],
                    f"TAT @Q={qa}": ra["tat"], f"TAT @Q={qb}": rb["tat"],
                    f"WT @Q={qa}": ra["wt"], f"WT @Q={qb}": rb["wt"],
                    f"RT @Q={qa}": ra["rt"], f"RT @Q={qb}": rb["rt"]})
    show_df(pd.DataFrame(per))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average times (ms)**")
        st.bar_chart(pd.DataFrame({f"Q={qa}": [ma["avg_tat"], ma["avg_wt"], ma["avg_rt"]],
                                   f"Q={qb}": [mb["avg_tat"], mb["avg_wt"], mb["avg_rt"]]},
                                  index=["Avg TAT", "Avg WT", "Avg RT"]))
    with c2:
        st.markdown("**Response time per process (ms)**")
        st.bar_chart(pd.DataFrame({f"Q={qa}": [r["rt"] for r in a["rows"]],
                                   f"Q={qb}": [r["rt"] for r in b["rows"]]},
                                  index=[r["pid"] for r in a["rows"]]))

    st.markdown("##### ✅ Quantum recommendation (generated from the numbers above)")
    if with_overhead:
        limit = st.number_input("Longest response time the emergency services can tolerate (ms)",
                                min_value=0, max_value=1000, value=10, step=1, key=f"{key}_limit")
        ok = [r for r in (a, b) if r["m"]["max_rt"] <= limit]
        if ok:
            best = min(ok, key=lambda r: (r["m"]["overhead"], r["m"]["avg_wt"]))
            other = b if best is a else a
            msg = (f"**Choose Q = {best['q']} ms.** Its worst-case response time is "
                   f"{best['m']['max_rt']} ms (limit {limit} ms) and it wastes only "
                   f"{best['m']['overhead']} ms on {best['m']['switches']} context switches "
                   f"(CPU efficiency {best['m']['efficiency']:.2f} %).")
            if other["m"]["max_rt"] > limit:
                msg += (f" Q = {other['q']} ms is rejected because its worst-case response time is "
                        f"{other['m']['max_rt']} ms, above the limit.")
            else:
                msg += (f" Q = {other['q']} ms also meets the limit but wastes "
                        f"{other['m']['overhead']} ms on context switching.")
            st.success(msg)
        else:
            best = min((a, b), key=lambda r: r["m"]["max_rt"])
            st.warning(f"Neither quantum keeps the worst-case response time within {limit} ms. "
                       f"Q = {best['q']} ms comes closest ({best['m']['max_rt']} ms) - "
                       f"try a smaller quantum or accept a longer limit.")
        st.caption("A smaller quantum answers every task sooner but adds more context-switch overhead; "
                   "a larger quantum cuts overhead but makes later processes wait longer for their first turn. "
                   "Change the limit above to see how the choice shifts.")
    else:
        criteria = [("average waiting time", "avg_wt"),
                    ("average response time (fair CPU access)", "avg_rt"),
                    ("number of context switches", "switches")]
        wa, wb = [], []
        for name, k in criteria:
            va, vb = round(ma[k], 6), round(mb[k], 6)
            if va < vb:
                wa.append(name)
            elif vb < va:
                wb.append(name)
        if len(wa) == len(wb):
            winner, loser = (a, b) if ma["avg_wt"] <= mb["avg_wt"] else (b, a)
            wins_w, wins_l = (wa, wb) if winner is a else (wb, wa)
            tie_note = " The criteria are evenly split, so the lower average waiting time breaks the tie."
        else:
            winner, loser = (a, b) if len(wa) > len(wb) else (b, a)
            wins_w, wins_l = (wa, wb) if winner is a else (wb, wa)
            tie_note = ""
        msg = f"**Suggested quantum: Q = {winner['q']} ms.** It is better on " + \
              (", ".join(wins_w) if wins_w else "no criterion") + "."
        if wins_l:
            msg += f" Q = {loser['q']} ms is better on " + ", ".join(wins_l) + "."
        st.success(msg + tie_note)
        st.caption("Fair CPU access is judged by response time (how soon every process gets its first turn); "
                   "context switches show the scheduling cost. Use the table above to justify your final choice.")


def activity_ui(key, heading, scenario, defaults, quanta, with_overhead):
    st.markdown(f"### {heading}")
    st.caption(scenario)

    with st.expander("1️⃣  Enter the workload (processes, arrival times, burst times)", expanded=True):
        procs = read_processes(key, defaults)
        c1, c2, c3 = st.columns(3)
        q1 = int(c1.number_input("Time quantum - run 1 (ms)", 1, 100, quanta[0], 1, key=f"{key}_q1"))
        q2 = int(c2.number_input("Time quantum - run 2 (ms)", 1, 100, quanta[1], 1, key=f"{key}_q2"))
        if with_overhead:
            cs = int(c3.number_input("Context-switch overhead (ms)", 1, 10, 1, 1, key=f"{key}_cs"))
        else:
            cs = 0
            c3.markdown("Context-switch overhead:  \n**ignored (0 ms)**")

    if sum(p["bt"] for p in procs) > 2000:
        st.error("Total burst time is too large for the visualizer (limit 2000 ms). Please reduce it.")
        return

    ra = simulate_rr(procs, q1, cs)
    rb = simulate_rr(procs, q2, cs)

    st.markdown("### 2️⃣  Simulation results")
    tabs = st.tabs([f"Quantum = {q1} ms", f"Quantum = {q2} ms", "📊 Comparison"])
    with tabs[0]:
        show_run(ra, with_overhead)
    with tabs[1]:
        show_run(rb, with_overhead)
    with tabs[2]:
        if q1 == q2:
            st.info("Both runs use the same quantum - choose two different values to compare them.")
        else:
            show_comparison(ra, rb, with_overhead, key)


# ════════════════════════════════════════════════════════════════════════════
#  4.  PAGE
# ════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="SDG Round Robin Visualizer", page_icon="🌍", layout="wide")
    st.title("🌍 SDG-Based Round Robin Scheduling Visualizer")
    st.caption("Operating Systems - Assignment 1 · Gantt charts, performance values and quantum comparison, "
               "with a step-by-step animated scheduler.")

    with st.expander("ℹ️  How the simulation works / assumptions"):
        st.markdown(
            """
- **Round Robin**: ready processes take turns on the CPU for at most one *time quantum*; an unfinished process goes to the back of the queue.
- If a process **arrives at the exact moment a slice ends**, it joins the queue *before* the pre-empted process (standard convention).
- **Completion time (CT)** = time the process finishes · **Turnaround (TAT)** = CT − AT · **Waiting (WT)** = TAT − BT · **Response (RT)** = first CPU time − AT.
- **Context switch** = the CPU moves from one process to a *different* process. Nothing is charged when the same process simply continues, before the first process, after the last process, or after an idle gap.
- **Activity 2** adds the overhead as its own interval in the Gantt chart (grey striped **CS** block).
- **CPU efficiency** = total burst time ÷ (total burst time + total context-switch overhead) × 100.
            """
        )

    t1, t2 = st.tabs(["🅰️ Activity 1 - Without context-switch overhead",
                      "🅱️ Activity 2 - With context-switch overhead"])
    with t1:
        activity_ui(
            "a1",
            "Activity 1 - Smart municipal control centre (SDG 6, 11, 12)",
            "Round Robin, context-switching overhead ignored. Compare two time quanta (default 2 ms vs 3 ms).",
            ACT1_DEFAULT, (2, 3), with_overhead=False,
        )
    with t2:
        activity_ui(
            "a2",
            "Activity 2 - Climate emergency response centre (SDG 11, 13)",
            "Round Robin with a context-switch overhead whenever the CPU changes process "
            "(default 1 ms). Compare two time quanta (default 2 ms vs 4 ms).",
            ACT2_DEFAULT, (2, 4), with_overhead=True,
        )


if __name__ == "__main__":
    main()