# RF-CL: Continual Learning for RF Modulation Classification Under Evolving SNR Conditions

**RF-CL** is a domain-incremental continual learning framework for **Radio Frequency (RF) modulation classification** under changing Signal-to-Noise Ratio (SNR) conditions.

The project studies how a neural network trained sequentially on different SNR domains can retain knowledge from previously encountered domains while learning from new ones. The project compares standard sequential learning with replay-based continual learning, including **Random Experience Replay** and **SNR-Aware Experience Replay**.

---

## 1. Project Overview

Automatic Modulation Classification (AMC) identifies the modulation scheme used by a received RF signal.

In a conventional machine learning setup, training and testing data are assumed to come from a relatively stable distribution. However, real-world RF environments can change over time because of channel conditions, interference, propagation effects, and changing SNR.

This creates a **continual learning** problem.

Instead of training on all SNR conditions simultaneously, RF-CL presents the model with a sequence of SNR domains:

```text
T1 → T2 → T3 → T4 → T5
```

The model must learn each new domain while retaining knowledge of previously encountered domains.

### Core research question

> Can replay-based continual learning reduce catastrophic forgetting when an RF modulation classifier encounters progressively more difficult SNR conditions?

---

# 2. Dataset

The project uses the **RadioML 2016.10a** dataset.

### Dataset characteristics

| Property           |            Value |
| ------------------ | ---------------: |
| Modulation classes |               11 |
| SNR values         |               20 |
| SNR range          | -20 dB to +18 dB |
| SNR step           |             2 dB |
| Samples per signal |   128 IQ samples |
| Input shape        |       `[2, 128]` |
| Channel 0          |     In-phase (I) |
| Channel 1          |   Quadrature (Q) |
| Total samples      |          220,000 |

The 11 modulation classes are:

```text
8PSK
AM-DSB
AM-SSB
BPSK
CPFSK
GFSK
PAM4
QAM16
QAM64
QPSK
WBFM
```

---

# 3. Continual Learning Setup

RF-CL uses **domain-incremental continual learning**.

The label space remains fixed across all tasks:

```text
11 modulation classes
```

Only the input domain changes, specifically the SNR range.

## Five sequential tasks

| Task | SNR Domain            |
| ---- | --------------------- |
| T1   | 12, 14, 16, 18 dB     |
| T2   | 4, 6, 8, 10 dB        |
| T3   | -4, -2, 0, 2 dB       |
| T4   | -12, -10, -8, -6 dB   |
| T5   | -20, -18, -16, -14 dB |

The model therefore experiences progressively more challenging signal conditions.

```text
High SNR
   │
   ▼
T1: 12 → 18 dB
   │
   ▼
T2:  4 → 10 dB
   │
   ▼
T3: -4 →  2 dB
   │
   ▼
T4: -12 → -6 dB
   │
   ▼
T5: -20 → -14 dB
   │
   ▼
Low SNR
```

---

# 4. Data Splitting

A fixed **70/15/15 split** is generated once and reused across experiments.

```text
70% → Training
15% → Validation
15% → Testing
```

For every task:

```text
Training   = 30,800 samples
Validation =  6,600 samples
Testing    =  6,600 samples
```

Across all five tasks:

```text
Training   = 154,000
Validation =  33,000
Testing    =  33,000
```

The test set is strictly isolated.

It is never used for:

* Training
* Replay memory
* Hyperparameter tuning
* Early stopping
* Model selection
* Threshold selection
* Debugging decisions based on performance

---

# 5. Data Contract

Each dataset sample follows the common project contract:

```python
{
    "x": tensor,   # shape [2, 128]
    "y": int,      # class label 0–10
    "snr": int,    # SNR value
    "task": int    # task ID 1–5
}
```

This contract allows the continual learning and evaluation modules to remain independent of the underlying model architecture.

---

# 6. Continual Learning Methods

RF-CL evaluates sequential learning with replay-based approaches.

## 6.1 Naive Sequential Learning

The model learns each task sequentially without accessing previous training data.

```text
T1 → Train
      ↓
T2 → Train
      ↓
T3 → Train
      ↓
T4 → Train
      ↓
T5 → Train
```

This provides the continual-learning baseline and exposes the effects of catastrophic forgetting.

---

## 6.2 Random Experience Replay

Random Replay maintains a fixed memory containing training samples from previously completed tasks.

When a new task arrives:

```text
Current Task Data
       +
Previous Memory
       ↓
Training
       ↓
Updated Memory
```

Samples from the available previous-task memory are selected randomly.

The replay memory contains only samples from tasks that have already been encountered.

Future tasks are never stored or accessed.

---

## 6.3 SNR-Aware Experience Replay

The primary replay method is **SNR-Aware Experience Replay**.

The method uses metadata already available in the dataset:

```text
Task
SNR
Modulation Class
```

Instead of treating every previous sample identically, the replay memory attempts to preserve diversity across:

* Previous tasks
* SNR conditions
* Modulation classes

Samples are organized into `(task, SNR, class)` groups and selected using balanced quotas and deterministic round-robin selection.

This helps prevent the memory from becoming dominated by a small subset of previously encountered RF conditions.

### Important distinction

The project does **not** claim that replay itself is novel.

The methodological focus is:

> **SNR-aware experience replay for RF domain-incremental learning.**

---

# 7. Replay Memory

The replay memory has a fixed capacity.

Default configuration:

```yaml
memory:
  size: 2000
```

The memory stores:

```text
x
y
snr
task
```

Only training samples from completed tasks can enter the memory.

The memory is never populated using validation or test samples.

### Memory evolution

```text
After T1:
Memory ← T1 training samples

After T2:
Memory ← T1 + T2 training samples

After T3:
Memory ← T1 + T2 + T3 training samples

After T4:
Memory ← T1 + T2 + T3 + T4 training samples

After T5:
Memory ← T1 + T2 + T3 + T4 + T5 training samples
```

subject to the fixed memory capacity.

---

# 8. Evaluation Protocol

Evaluation is performed after every task.

Only test data from tasks that have already been encountered may be evaluated.

The resulting accuracy matrix is:

```text
             Evaluated Task
             T1    T2    T3    T4    T5

After T1     A11
After T2     A21   A22
After T3     A31   A32   A33
After T4     A41   A42   A43   A44
After T5     A51   A52   A53   A54   A55
```

This allows us to distinguish:

* Learning the current task
* Retaining previous tasks
* Forgetting previously learned domains

Future-task test data is never evaluated before that task has been learned.

---

# 9. Evaluation Metrics

## 9.1 Accuracy

Standard classification accuracy is calculated for each task.

---

## 9.2 Macro F1

Macro F1 calculates the F1 score independently for each modulation class and averages them.

This gives every class equal importance, which is useful for evaluating performance across the 11 modulation classes.

---

## 9.3 Per-Class Accuracy

Accuracy is calculated separately for every modulation class.

This helps identify classes that are particularly difficult under low-SNR conditions.

---

## 9.4 Confusion Matrix

Confusion matrices show which modulation classes are being confused with one another.

They are especially useful for analyzing classification degradation as SNR decreases.

---

# 10. Average Accuracy

The primary final performance metric is **Average Accuracy (AA)**.

After T5, the accuracy of each of the five task domains is averaged:

```text
AA = mean(A51, A52, A53, A54, A55)
```

Higher AA indicates better overall retention and final classification performance across all encountered domains.

---

# 11. Average Forgetting

Forgetting is measured for tasks T1–T4.

For each previous task:

```text
F1 = A11 - A51
F2 = A22 - A52
F3 = A33 - A53
F4 = A44 - A54
```

Average Forgetting is:

```text
AF = (F1 + F2 + F3 + F4) / 4
```

T5 is excluded because there is no later task after T5 against which forgetting can be measured.

Lower AF indicates better retention.

---

# 12. Reproducibility

All experiments use:

```text
Seed = 42
```

The same:

* Dataset
* Task sequence
* Train/validation/test splits
* Model initialization
* Optimizer
* Learning rate
* Batch size
* Number of epochs
* Memory capacity

must be used when comparing Random Replay and SNR-Aware Replay.

The intended comparison changes only the **memory selection strategy**.

---

# 13. Project Structure

```text
rf-continual-learning/
│
├── config.yaml
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── RML2016.10a_dict.pkl
│   │
│   └── splits/
│       └── splits_seed42.json
│
├── src/
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── prepare_data.py
│   │   ├── raw_loader.py
│   │   ├── splits.py
│   │   ├── synthetic.py
│   │   ├── tasks.py
│   │   └── __init__.py
│   │
│   ├── continual/
│   │   ├── memory.py
│   │   ├── replay.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── plots.py
│   │
│   ├── models/
│   │   └── ...
│   │
│   ├── training/
│   │   └── ...
│   │
│   └── utils/
│
├── experiments/
│   └── ...
│
├── tests/
│   ├── test_continual.py
│   └── test_metrics.py
│
└── results/
    ├── metrics/
    └── figures/
```

---

# 14. Installation

## Requirements

Recommended environment:

```text
Python 3.13
PyTorch
NumPy
scikit-learn
PyYAML
matplotlib
pytest
```

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 15. Dataset Setup

Place the RadioML 2016.10a dataset at:

```text
data/raw/RML2016.10a_dict.pkl
```

The raw dataset is intentionally excluded from Git because of its large file size.

The repository expects the file locally.

---

# 16. Generate Fixed Dataset Splits

After placing the raw dataset in `data/raw/`, generate the fixed splits:

```powershell
python src/data/prepare_data.py
```

The generated split file is:

```text
data/splits/splits_seed42.json
```

The preparation process verifies:

* Five task domains
* Correct SNR assignments
* 70/15/15 splitting
* Class distribution
* No train/validation/test overlap
* No cross-task overlap

---

# 17. Verify the Dataset

Check all five task sizes:

```powershell
python -c "from src.data import get_task_dataset; [print('T%d: train=%d, val=%d, test=%d' % (t, len(get_task_dataset(t, 'train')), len(get_task_dataset(t, 'val')), len(get_task_dataset(t, 'test')))) for t in range(1, 6)]"
```

Expected:

```text
T1: train=30800, val=6600, test=6600
T2: train=30800, val=6600, test=6600
T3: train=30800, val=6600, test=6600
T4: train=30800, val=6600, test=6600
T5: train=30800, val=6600, test=6600
```

---

# 18. Run Tests

Run the continual learning and metric tests:

```powershell
python -m pytest tests/test_continual.py tests/test_metrics.py -v
```

The current Person 3 test suite covers:

### Replay memory

* Empty memory
* Valid sample insertion
* Capacity limits
* Metadata preservation
* Invalid sample shape
* Invalid class labels

### Random Replay

* Seen-task restriction
* Future-task exclusion
* Deterministic replay

### SNR-Aware Replay

* Capacity handling
* Metadata preservation
* Deterministic selection
* Seen-task restriction

### Evaluation

* Accuracy
* Macro F1
* Zero-division handling
* Per-class accuracy
* Confusion matrices
* Average Accuracy
* Average Forgetting
* Incomplete accuracy-matrix validation

Current validation:

```text
23 tests passed
```

---

# 19. Configuration

The main configuration is stored in:

```text
config.yaml
```

Current task configuration:

```yaml
seed: 42

dataset:
  name: RadioML2016.10a
  num_classes: 11
  input_length: 128

split:
  train: 0.70
  validation: 0.15
  test: 0.15

tasks:
  task_1:
    snr: [12, 14, 16, 18]

  task_2:
    snr: [4, 6, 8, 10]

  task_3:
    snr: [-4, -2, 0, 2]

  task_4:
    snr: [-12, -10, -8, -6]

  task_5:
    snr: [-20, -18, -16, -14]

training:
  batch_size: 128
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001

replay:
  memory_size: 2000
  methods:
    - random
    - snr_aware
```

---

# 20. Results

Experiment outputs are stored under:

```text
results/
├── metrics/
└── figures/
```

Expected outputs include:

```text
Accuracy matrix
Average Accuracy
Average Forgetting
Per-task accuracy
Macro F1
Per-class accuracy
Confusion matrices
```

The project should report results only from the real RadioML dataset.

Synthetic data is used exclusively for software/unit testing and must not be presented as experimental research results.

---

# 21. Experimental Comparison

The main comparison is:

```text
                 Sequential RF Learning
                         │
              ┌──────────┴──────────┐
              │                     │
         Random Replay        SNR-Aware Replay
              │                     │
              └──────────┬──────────┘
                         ↓
                   Evaluation
                         ↓
                  AA + AF + F1
```

Both replay methods should use identical experimental conditions.

### Controlled variables

```text
Dataset             Same
Task sequence       Same
Splits               Same
Seed                 Same
Model                Same
Initialization      Same
Optimizer            Same
Learning rate       Same
Batch size           Same
Epochs               Same
Memory capacity     Same
```

The only intended difference is the replay-memory selection strategy.

---

# 22. Ablation Study

The project can evaluate which aspects of SNR-aware replay contribute to performance.

Possible ablations include:

```text
Full SNR-aware replay
        ↓
Remove SNR balancing
        ↓
Remove class balancing
        ↓
Remove task balancing
        ↓
Random replay
```

The exact ablation configuration should be implemented consistently with the final experimental protocol.

---

# 23. Important Experimental Rules

To maintain a valid continual-learning experiment:

### Do

* Use only current and previously observed training data.
* Keep test data isolated.
* Use fixed splits across methods.
* Use the same random seed.
* Evaluate after every task.
* Track the complete accuracy matrix.
* Report both Average Accuracy and Average Forgetting.
* Preserve task, SNR, and class metadata in replay memory.
* Keep the model architecture unchanged between replay methods.

### Do not

* Replay test samples.
* Train on future tasks.
* Tune using test accuracy.
* Change hyperparameters based on test results.
* Select the best model using the final test set.
* Claim replay itself as the novel contribution.
* Compare methods using different splits or initialization conditions.
* Report synthetic test results as research results.

---

# 24. Design Philosophy

The project follows a modular architecture.

```text
                ┌──────────────────┐
                │    RadioML Data  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Task Definition │
                │      T1 → T5     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      Model       │
                └────────┬─────────┘
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
             No Replay       Replay
                                │
                       ┌────────┴────────┐
                       │                 │
                       ▼                 ▼
                  Random Replay    SNR-Aware Replay
                       │                 │
                       └────────┬────────┘
                                ▼
                         Evaluation
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                   AA          AF        Macro F1
                                │
                                ▼
                         Results + Figures
```

The model is treated as a black box by the continual-learning and evaluation modules:

```python
logits = model(x)
```

This keeps the continual-learning implementation independent of the specific neural network architecture.

---

# 25. Reproducibility Checklist

Before reporting an experiment, verify:

```text
[ ] RadioML 2016.10a is used
[ ] Five SNR tasks are correct
[ ] Seed = 42
[ ] Fixed 70/15/15 splits are used
[ ] Same splits are used for every method
[ ] Test data is isolated
[ ] Memory contains training samples only
[ ] Memory capacity is fixed
[ ] Random Replay and SNR-Aware Replay use identical training settings
[ ] Evaluation occurs after every task
[ ] Accuracy matrix is complete
[ ] Average Accuracy is calculated from T5
[ ] Average Forgetting uses T1–T4
[ ] Macro F1 is reported
[ ] Per-class metrics are available
[ ] Confusion matrices are generated
[ ] Synthetic data is not used for reported results
```

---

# 26. Current Development Status

| Component                | Status                            |
| ------------------------ | --------------------------------- |
| RadioML data pipeline    | Complete                          |
| Five-task SNR definition | Complete                          |
| Fixed dataset splits     | Complete                          |
| Replay memory            | Complete                          |
| Random Replay            | Complete                          |
| SNR-Aware Replay         | Complete                          |
| Evaluation metrics       | Complete                          |
| Accuracy matrix          | Complete                          |
| Forgetting calculation   | Complete                          |
| Plotting utilities       | Complete                          |
| Person 3 unit tests      | Complete                          |
| Model implementation     | In progress                       |
| Training integration     | In progress                       |
| Full replay experiments  | Pending model/trainer integration |
| Final results            | Pending experiments               |
| Final ablation results   | Pending experiments               |

---

# 27. Research Contribution

The project investigates continual learning for RF modulation classification under **evolving SNR domains**.

The primary methodological contribution is:

> **SNR-aware experience replay for RF domain-incremental learning.**

The approach leverages SNR and modulation metadata to maintain a more diverse replay memory across previously encountered RF conditions.

The project evaluates whether this strategy can:

1. Improve retention of previously learned SNR domains.
2. Reduce catastrophic forgetting.
3. Maintain modulation-class performance as SNR conditions evolve.
4. Improve final Average Accuracy compared with sequential learning and Random Replay.

---

# 28. Team Responsibilities

### Person 1: Data Pipeline

Responsible for:

* RadioML loading
* Dataset representation
* Task construction
* Fixed train/validation/test splits
* Data integrity

### Person 2: Model & Training

Responsible for:

* Neural network architecture
* Baseline training
* Training interface
* Optimization

### Person 3: Continual Learning, Memory & Evaluation

Responsible for:

* Replay memory
* Random Replay
* SNR-Aware Replay
* Continual-learning experiment logic
* Evaluation
* Metrics
* Accuracy matrix
* Forgetting
* Plotting
* Ablation studies
* Continual-learning tests

---

# 29. License and Dataset Notice

This repository contains code for research and educational purposes.

The RadioML 2016.10a dataset is not included in the repository because of its size and distribution requirements. Users should obtain and use the dataset according to its applicable terms.

---

# 30. Summary

RF-CL models RF modulation classification as a sequence of changing SNR domains rather than as a single stationary classification problem.

The core sequence is:

```text
RadioML 2016.10a
       ↓
Fixed 5-task SNR sequence
       ↓
Sequential training
       ↓
Replay memory
       ↓
Random / SNR-Aware Replay
       ↓
Evaluation after every task
       ↓
Accuracy Matrix
       ↓
Average Accuracy + Forgetting
       ↓
Analysis of catastrophic forgetting
```

The central goal is to determine whether **SNR-aware experience replay can preserve previously learned RF modulation knowledge more effectively under evolving SNR conditions**.
