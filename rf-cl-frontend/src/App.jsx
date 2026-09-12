import React, { useMemo, useState } from "react";
import { tasks, methods, modulationClasses } from "./data";

const pct = (x) => `${(x * 100).toFixed(1)}%`;

function Metric({ label, value, sub, accent }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span><i className={accent || ""} />
      </div>
      <strong>{value}</strong>
      {sub && <small>{sub}</small>}
    </div>
  );
}

function Heatmap({ method }) {
  const m = methods[method];
  return (
    <div className="heatmap-wrap">
      <div className="heatmap-header">
        <span></span>
        {tasks.map(t => <span key={t.id}>T{t.id}</span>)}
      </div>
      {tasks.map((rowTask, r) => (
        <div className="heatmap-row" key={r}>
          <span className="row-label">After T{rowTask.id}</span>
          {tasks.map((_, c) => {
            const v = m.matrix?.[r]?.[c];
            return (
              <div
                key={c}
                className={`heat-cell ${v == null ? "empty" : ""}`}
                style={v == null ? {} : {"--v": v}}
                title={v == null ? "Not evaluated yet" : `After T${r+1}, Test T${c+1}: ${pct(v)}`}
              >
                {v == null ? "·" : pct(v)}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function BarChart({ values, max=1 }) {
  return (
    <div className="bars">
      {values.map(({label, value, cls}) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div className="bar-track"><div className={`bar ${cls || ""}`} style={{width: `${Math.max(3, value/max*100)}%`}} /></div>
          <b>{pct(value)}</b>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [selected, setSelected] = useState("SNR-Aware Replay");
  const [tab, setTab] = useState("overview");

  const final = methods[selected].matrix?.[4] || [];
  const finalAA = methods[selected].aa ?? final.reduce((a,b)=>a+b,0) / (final.length || 1);
  const finalAF = methods[selected].af;

  const comparisonAA = useMemo(() => [
    {label:"Naive", value: methods["Naive Sequential"].matrix[4].reduce((a,b)=>a+b,0)/5, cls:"cyan"},
    {label:"Random Replay", value: methods["Random Replay"].aa, cls:"violet"},
    {label:"SNR-Aware", value: methods["SNR-Aware Replay"].aa, cls:"lime"},
  ], []);

  const comparisonAF = [
    {label:"Naive", value: (methods["Naive Sequential"].matrix[0][0]-methods["Naive Sequential"].matrix[4][0] +
      methods["Naive Sequential"].matrix[1][1]-methods["Naive Sequential"].matrix[4][1] +
      methods["Naive Sequential"].matrix[2][2]-methods["Naive Sequential"].matrix[4][2] +
      methods["Naive Sequential"].matrix[3][3]-methods["Naive Sequential"].matrix[4][3])/4, cls:"cyan"},
    {label:"Random Replay", value: methods["Random Replay"].af, cls:"violet"},
    {label:"SNR-Aware", value: methods["SNR-Aware Replay"].af, cls:"lime"},
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">⌁</span><div><b>RF-CL</b><small>Research Dashboard</small></div></div>
        <nav>
          {[
            ["overview","Overview","◈"],
            ["tasks","Task Progress","◫"],
            ["replay","Replay Memory","◉"],
            ["evaluation","Evaluation","▦"],
          ].map(([id,label,icon]) =>
            <button className={tab===id?"active":""} onClick={()=>setTab(id)} key={id}><span>{icon}</span>{label}</button>
          )}
        </nav>
        <div className="side-status"><span className="pulse"></span><div><b>Experiment</b><small>Baseline run active</small></div></div>
        <div className="sidebar-foot">RadioML 2016.10a<br/>Seed 42 · PyTorch</div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <div className="eyebrow">CONTINUAL LEARNING · RF MODULATION</div>
            <h1>RF-CL <span>/</span> Experiment Control</h1>
          </div>
          <div className="top-actions"><span className="chip"><i className="dot green"></i> 5 domains</span><span className="chip">CPU · 55,691 params</span></div>
        </header>

        {tab === "overview" && <>
          <section className="hero">
            <div>
              <span className="status-pill">● RUN COMPLETE · REPLAY</span>
              <h2>Learning while the RF world keeps changing.</h2>
              <p>Domain-incremental modulation classification across five sequential SNR conditions, with replay memory designed to resist catastrophic forgetting.</p>
            </div>
            <div className="hero-orbit"><div className="orbit o1"></div><div className="orbit o2"></div><div className="orbit-core">RF</div></div>
          </section>

          <section className="metrics-grid">
            <Metric label="SNR-Aware AA" value="49.3%" sub="Average Accuracy · T5" accent="lime" />
            <Metric label="SNR-Aware AF" value="12.9%" sub="Average Forgetting · lower is better" accent="lime" />
            <Metric label="Replay Memory" value="2,000" sub="training samples" accent="violet" />
            <Metric label="Task Domains" value="05" sub="4 SNRs per domain" accent="cyan" />
          </section>

          <section className="section-head"><div><span className="eyebrow">METHOD COMPARISON</span><h3>What survives the task sequence?</h3></div>
            <div className="method-select">
              {Object.keys(methods).map(name => <button className={selected===name?"selected":""} onClick={()=>setSelected(name)} key={name}>{name}</button>)}
            </div>
          </section>

          <section className="dashboard-grid">
            <div className="panel large">
              <div className="panel-head"><div><b>Accuracy Matrix</b><small>Rows = training endpoint · columns = seen test task</small></div><span className="legend"><i className={methods[selected].color}></i>{selected}</span></div>
              {methods[selected].pending ? <div className="pending">Joint training is still running in the terminal.<br/><small>Its matrix will appear here when the run finishes.</small></div> : <Heatmap method={selected}/>}
            </div>

            <div className="panel">
              <div className="panel-head"><div><b>Final Task Accuracy</b><small>Performance after T5</small></div></div>
              {final.length ? <BarChart values={final.map((v,i)=>({label:`T${i+1}`,value:v,cls:methods[selected].color}))} /> : <div className="pending">Waiting for joint results</div>}
            </div>

            <div className="panel">
              <div className="panel-head"><div><b>Average Accuracy</b><small>Final T5 mean across tasks</small></div></div>
              <BarChart values={comparisonAA} />
              <div className="callout"><b>+5.9 pp</b><span>SNR-Aware vs Random Replay</span></div>
            </div>

            <div className="panel">
              <div className="panel-head"><div><b>Average Forgetting</b><small>Lower is better</small></div></div>
              <BarChart values={comparisonAF} max={0.25}/>
              <div className="callout"><b>−6.7 pp</b><span>Forgetting reduction vs Random Replay</span></div>
            </div>
          </section>
        </>}

        {tab === "tasks" && <Page title="Task Progress" kicker="SEQUENTIAL DOMAINS">
          <div className="task-flow">{tasks.map((t,i)=><React.Fragment key={t.id}><div className="task-card"><span>T{t.id}</span><b>{t.snr}</b><small>SNR domain · {t.values.length} SNR levels</small><div className="snr-pills">{t.values.map(v=><em key={v}>{v > 0 ? "+" : ""}{v}</em>)}</div></div>{i<4 && <div className="arrow">→</div>}</React.Fragment>)}</div>
          <div className="panel full"><div className="panel-head"><div><b>Fixed experimental protocol</b><small>Same split and label space across every method</small></div></div><div className="protocol"><div><b>70%</b><span>Train</span></div><div><b>15%</b><span>Validation</span></div><div><b>15%</b><span>Test</span></div><div><b>11</b><span>Classes</span></div><div><b>42</b><span>Seed</span></div></div></div>
        </Page>}

        {tab === "replay" && <Page title="Replay Memory" kicker="EXPERIENCE REPLAY">
          <div className="replay-diagram"><div className="node"><b>Current Task</b><span>30,800 samples</span></div><div className="connector">＋</div><div className="node violet-node"><b>Previous Memory</b><span>up to 2,000 samples</span></div><div className="connector">→</div><div className="node lime-node"><b>Training Batch</b><span>current + replay</span></div><div className="connector">→</div><div className="node"><b>Updated Memory</b><span>2,000 retained</span></div></div>
          <div className="comparison-cards"><div className="panel"><b>Random Replay</b><p>Uniform random selection from eligible historical samples.</p><span className="tag violet">BASELINE</span></div><div className="panel highlight"><b>SNR-Aware Replay</b><p>Round-robin selection across task, SNR and class groups to preserve domain diversity.</p><span className="tag lime">PRIMARY METHOD</span></div></div>
        </Page>}

        {tab === "evaluation" && <Page title="Evaluation" kicker="RESEARCH METRICS">
          <div className="evaluation-list">{[
            ["Accuracy","Overall classification correctness on each seen task."],
            ["Macro F1","Class-balanced F1, useful when per-class performance differs."],
            ["Per-class Accuracy","Accuracy broken down across all 11 modulation classes."],
            ["Confusion Matrix","11 × 11 class confusion after each task."],
            ["Average Accuracy","Mean of final accuracies A51...A55."],
            ["Average Forgetting","Mean drop for T1...T4 between first learning and T5."]
          ].map(([a,b])=><div className="eval-item" key={a}><span>◌</span><div><b>{a}</b><small>{b}</small></div></div>)}</div>
        </Page>}

        <footer>RF-CL · Continual Learning for RF Modulation Classification Under Evolving SNR Conditions</footer>
      </main>
    </div>
  );
}
function Page({title,kicker,children}) {
  return <><section className="page-title"><span className="eyebrow">{kicker}</span><h2>{title}</h2><p>RadioML 2016.10a · 11 classes · 5 sequential SNR domains</p></section>{children}</>;
}

export default App;