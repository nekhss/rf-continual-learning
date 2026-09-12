import React, { useState } from "react";
import { tasks, methods } from "./data";

const pct = (x) => `${(x * 100).toFixed(1)}%`;

function Metric({ label, value, sub }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <i />
      </div>
      <strong>{value}</strong>
      <small>{sub}</small>
    </div>
  );
}

function Heatmap({ matrix }) {
  return (
    <div className="heatmap-wrap">
      <div className="heatmap-header">
        <span />
        {tasks.map((t) => (
          <span key={t.id}>T{t.id}</span>
        ))}
      </div>

      {tasks.map((task, row) => (
        <div className="heatmap-row" key={task.id}>
          <span className="row-label">After T{task.id}</span>

          {tasks.map((_, col) => {
            const value = matrix[row]?.[col];

            return (
              <div
                className={`heat-cell ${value == null ? "empty" : ""}`}
                key={col}
                style={
                  value == null
                    ? {}
                    : {
                        "--v": value,
                      }
                }
              >
                {value == null ? "·" : pct(value)}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function BarChart({ values, max = 1 }) {
  return (
    <div className="bars">
      {values.map(({ label, value, cls = "" }) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>

          <div className="bar-track">
            <div
              className={`bar ${cls}`}
              style={{
                width: `${Math.max(3, (value / max) * 100)}%`,
              }}
            />
          </div>

          <b>{pct(value)}</b>
        </div>
      ))}
    </div>
  );
}

function Page({ title, kicker, children }) {
  return (
    <>
      <section className="page-title">
        <span className="eyebrow">{kicker}</span>
        <h2>{title}</h2>
        <p>
          RadioML 2016.10a · 11 classes · 5 sequential SNR domains
        </p>
      </section>

      {children}
    </>
  );
}

function App() {
  const [tab, setTab] = useState("overview");
  const [selected, setSelected] = useState("SNR-Aware Replay");

  const selectedMethod = methods[selected];
  const finalValues = selectedMethod.matrix?.[4] || [];

  const naiveFinal = methods["Naive Sequential"].matrix[4];

  const naiveAA =
    naiveFinal.reduce((sum, value) => sum + value, 0) /
    naiveFinal.length;

  const comparisonAA = [
    {
      label: "Naive",
      value: naiveAA,
      cls: "cyan",
    },
    {
      label: "Random Replay",
      value: methods["Random Replay"].aa,
      cls: "violet",
    },
    {
      label: "SNR-Aware",
      value: methods["SNR-Aware Replay"].aa,
      cls: "lime",
    },
  ];

  const comparisonAF = [
    {
      label: "Naive",
      value:
        (
          naiveFinal.reduce(
            (sum, value, index) =>
              sum +
              (methods["Naive Sequential"].matrix[index][index] -
                value),
            0
          ) / 4
        ),
      cls: "cyan",
    },
    {
      label: "Random Replay",
      value: methods["Random Replay"].af,
      cls: "violet",
    },
    {
      label: "SNR-Aware",
      value: methods["SNR-Aware Replay"].af,
      cls: "lime",
    },
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">⌁</span>

          <div>
            <b>RF-CL</b>
            <small>Research Dashboard</small>
          </div>
        </div>

        <nav>
          {[
            ["overview", "Overview", "◈"],
            ["tasks", "Task Progress", "◫"],
            ["replay", "Replay Memory", "◉"],
            ["evaluation", "Evaluation", "▦"],
          ].map(([id, label, icon]) => (
            <button
              key={id}
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}
            >
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>

        <div className="side-status">
          <span className="pulse" />

          <div>
            <b>Experiment</b>
            <small>Baseline run active</small>
          </div>
        </div>

        <div className="sidebar-foot">
          RadioML 2016.10a
          <br />
          Seed 42 · PyTorch
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <div className="eyebrow">
              CONTINUAL LEARNING · RF MODULATION
            </div>

            <h1>
              RF-CL <span>/</span> Experiment Control
            </h1>
          </div>

          <div className="top-actions">
            <span className="chip">
              <i className="dot green" />
              5 domains
            </span>

            <span className="chip">
              CPU · 55,691 params
            </span>
          </div>
        </header>

        {tab === "overview" && (
          <>
            <section className="hero">
              <div>
                <span className="status-pill">
                  ● RESEARCH DASHBOARD
                </span>

                <h2>
                  Learning while the RF world keeps changing.
                </h2>

                <p>
                  Domain-incremental modulation classification across
                  five sequential SNR conditions, with replay memory
                  designed to resist catastrophic forgetting.
                </p>
              </div>

              <div className="hero-orbit">
                <div className="orbit o1" />
                <div className="orbit o2" />

                <div className="orbit-core">RF</div>
              </div>
            </section>

            <section className="metrics-grid">
              <Metric
                label="SNR-Aware AA"
                value="49.3%"
                sub="Average Accuracy · T5"
              />

              <Metric
                label="SNR-Aware AF"
                value="12.9%"
                sub="Average Forgetting · lower is better"
              />

              <Metric
                label="Replay Memory"
                value="2,000"
                sub="training samples"
              />

              <Metric
                label="Task Domains"
                value="05"
                sub="4 SNR levels per domain"
              />
            </section>

            <section className="section-head">
              <div>
                <div className="eyebrow">METHOD COMPARISON</div>

                <h3>What survives the task sequence?</h3>
              </div>

              <div className="method-select">
                {Object.keys(methods).map((name) => (
                  <button
                    key={name}
                    className={
                      selected === name ? "selected" : ""
                    }
                    onClick={() => setSelected(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </section>

            <section className="dashboard-grid">
              <div className="panel large">
                <div className="panel-head">
                  <div>
                    <b>Accuracy Matrix</b>
                    <small>
                      Rows = training endpoint · columns = test task
                    </small>
                  </div>

                  <span className="legend">
                    <i className={selectedMethod.color} />
                    {selected}
                  </span>
                </div>

                {selectedMethod.pending ? (
                  <div className="pending">
                    Joint Training is still running.
                    <br />
                    <small>
                      Its results will appear here after the
                      experiment finishes.
                    </small>
                  </div>
                ) : (
                  <Heatmap matrix={selectedMethod.matrix} />
                )}
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div>
                    <b>Final Task Accuracy</b>
                    <small>Performance after T5</small>
                  </div>
                </div>

                {finalValues.length > 0 ? (
                  <BarChart
                    values={finalValues.map((value, index) => ({
                      label: `T${index + 1}`,
                      value,
                      cls: selectedMethod.color,
                    }))}
                  />
                ) : (
                  <div className="pending">
                    Waiting for Joint Training.
                  </div>
                )}
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div>
                    <b>Average Accuracy</b>
                    <small>
                      Final mean across all five tasks
                    </small>
                  </div>
                </div>

                <BarChart values={comparisonAA} />

                <div className="callout">
                  <b>+5.9 pp</b>
                  <span>
                    SNR-Aware vs Random Replay
                  </span>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div>
                    <b>Average Forgetting</b>
                    <small>Lower is better</small>
                  </div>
                </div>

                <BarChart values={comparisonAF} max={0.25} />

                <div className="callout">
                  <b>−6.7 pp</b>
                  <span>
                    Forgetting reduction vs Random Replay
                  </span>
                </div>
              </div>
            </section>
          </>
        )}

        {tab === "tasks" && (
          <Page
            title="Task Progress"
            kicker="SEQUENTIAL DOMAINS"
          >
            <div className="task-flow">
              {tasks.map((task, index) => (
                <React.Fragment key={task.id}>
                  <div className="task-card">
                    <span>T{task.id}</span>

                    <b>{task.snr}</b>

                    <small>
                      SNR domain · {task.values.length} levels
                    </small>

                    <div className="snr-pills">
                      {task.values.map((value) => (
                        <em key={value}>
                          {value > 0 ? "+" : ""}
                          {value}
                        </em>
                      ))}
                    </div>
                  </div>

                  {index < tasks.length - 1 && (
                    <div className="arrow">→</div>
                  )}
                </React.Fragment>
              ))}
            </div>

            <div className="panel full">
              <div className="panel-head">
                <div>
                  <b>Fixed Experimental Protocol</b>
                  <small>
                    Same split and label space across methods
                  </small>
                </div>
              </div>

              <div className="protocol">
                <div>
                  <b>70%</b>
                  <span>Train</span>
                </div>

                <div>
                  <b>15%</b>
                  <span>Validation</span>
                </div>

                <div>
                  <b>15%</b>
                  <span>Test</span>
                </div>

                <div>
                  <b>11</b>
                  <span>Classes</span>
                </div>

                <div>
                  <b>42</b>
                  <span>Seed</span>
                </div>
              </div>
            </div>
          </Page>
        )}

        {tab === "replay" && (
          <Page
            title="Replay Memory"
            kicker="EXPERIENCE REPLAY"
          >
            <div className="replay-diagram">
              <div className="node">
                <b>Current Task</b>
                <span>30,800 samples</span>
              </div>

              <div className="connector">＋</div>

              <div className="node violet-node">
                <b>Previous Memory</b>
                <span>up to 2,000 samples</span>
              </div>

              <div className="connector">→</div>

              <div className="node lime-node">
                <b>Training Batch</b>
                <span>current + replay</span>
              </div>

              <div className="connector">→</div>

              <div className="node">
                <b>Updated Memory</b>
                <span>2,000 retained</span>
              </div>
            </div>

            <div className="comparison-cards">
              <div className="panel">
                <b>Random Replay</b>

                <p>
                  Uniform random selection from eligible historical
                  training samples.
                </p>

                <span className="tag violet">
                  BASELINE
                </span>
              </div>

              <div className="panel highlight">
                <b>SNR-Aware Replay</b>

                <p>
                  Selection across task, SNR and class groups to
                  preserve domain diversity.
                </p>

                <span className="tag lime">
                  PRIMARY METHOD
                </span>
              </div>
            </div>
          </Page>
        )}

        {tab === "evaluation" && (
          <Page
            title="Evaluation"
            kicker="RESEARCH METRICS"
          >
            <div className="evaluation-list">
              {[
                [
                  "Accuracy",
                  "Overall classification correctness on each seen task.",
                ],
                [
                  "Macro F1",
                  "Class-balanced F1 across the 11 modulation classes.",
                ],
                [
                  "Per-class Accuracy",
                  "Accuracy reported separately for every modulation class.",
                ],
                [
                  "Confusion Matrix",
                  "11 × 11 class confusion after each task.",
                ],
                [
                  "Average Accuracy",
                  "Mean of final accuracies A51 through A55.",
                ],
                [
                  "Average Forgetting",
                  "Mean accuracy drop for T1 through T4 between first learning and T5.",
                ],
              ].map(([title, description]) => (
                <div className="eval-item" key={title}>
                  <span>◌</span>

                  <div>
                    <b>{title}</b>
                    <small>{description}</small>
                  </div>
                </div>
              ))}
            </div>
          </Page>
        )}

        <footer>
          RF-CL · Continual Learning for RF Modulation
          Classification Under Evolving SNR Conditions
        </footer>
      </main>
    </div>
  );
}

export default App;