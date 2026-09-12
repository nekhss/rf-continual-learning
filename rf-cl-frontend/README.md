# RF-CL Frontend

Research dashboard for **RF-CL: Continual Learning for RF Modulation Classification Under Evolving SNR Conditions**.

## Run

```bash
npm install
npm run dev
```

The dashboard currently includes the verified 10-epoch replay results and the completed Naive Sequential matrix. Joint Training is intentionally shown as pending until the current baseline run finishes.

## Data wiring

The first version keeps research results in `src/data.js` so the UI can be developed independently of the Python experiment runner. After the final experiment files are available, replace this adapter with a fetch/JSON loader for:

- `results/baselines_final/`
- `results/replay_final/`
- `results/figures/`

Do not move or alter the core continual-learning implementation just to support the UI.
