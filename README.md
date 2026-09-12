# RF-CL: Continual Learning for RF Modulation Classification Under Evolving SNR Conditions

> **Domain-Incremental Continual Learning for Radio-Frequency Modulation Classification**

RF-CL studies how a neural network for RF modulation classification behaves when the signal environment changes over time.

The project models **evolving Signal-to-Noise Ratio (SNR) conditions as sequential domains** and investigates whether experience replay can reduce catastrophic forgetting while maintaining performance on newly encountered domains.

The primary continual-learning method is **SNR-Aware Experience Replay**, which uses task, SNR, and class diversity when constructing a fixed-size replay memory.

---

## 1. Problem Statement

In a conventional supervised-learning setting, a model can be trained using data from all available conditions simultaneously.

In a continual-learning setting, data arrives sequentially.

For RF modulation classification, this can represent a system operating under changing signal-quality conditions:

```text
New RF environment
       ↓
New SNR domain
       ↓
Model learns new domain
       ↓
Previous knowledge may be forgotten
       ↓
Catastrophic Forgetting
```

RF-CL investigates whether replaying carefully selected samples from previously observed domains can improve knowledge retention.

### Research Question

> **Can SNR-aware experience replay improve retention of previously learned RF modulation domains under evolving SNR conditions compared with naive sequential learning and random replay?**

---

# 2. Project Formulation

RF-CL is formulated as **domain-incremental continual learning**.

The important distinction is:

* The **11 modulation classes remain fixed** across all tasks.
* The **SNR distribution changes between tasks**.
* The model therefore encounters new domains rather than new classes.

This is **not class-incremental learning**.

### Fixed label space

```text
11 modulation classes
        ↓
Available in every task
        ↓
Only the SNR domain changes
```

---

# 3. Dataset

## RadioML 2016.10a

The project uses the **RadioML 2016.10a** dataset.

### Dataset characteristics

| Property           |            Value |
| ------------------ | ---------------: |
| Modulation classes |               11 |
| SNR values         | -20 dB to +18 dB |
| SNR step           |             2 dB |
| IQ samples         |              128 |
| Input shape        |       `[2, 128]` |
| Channel 0          |     In-phase (I) |
| Channel 1          |   Quadrature (Q) |
| Random seed        |               42 |

The complete dataset contains **220,000 samples** across the 11 modulation classes and 20 SNR conditions.

---

# 4. Sequential Task Configuration

The complete SNR range is divided into five sequential domains.

| Task   | SNR values             |
| ------ | ---------------------- |
| **T1** | `[12, 14, 16, 18]`     |
| **T2** | `[4, 6, 8, 10]`        |
| **T3** | `[-4, -2, 0, 2]`       |
| **T4** | `[-12, -10, -8, -6]`   |
| **T5** | `[-20, -18, -16, -14]` |

The sequence therefore progresses from relatively high-SNR conditions toward increasingly noisy conditions.

```text
T1                 T2                 T3                 T4                 T5
+12 → +18          +4 → +10          -4 → +2           -12 → -6          -20 → -14
   │                  │                  │                  │                  │
   └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
                         Sequential domain evolution
```

Each task contains:

* 30,800 training samples
* 6,600 validation samples
* 6,600 test samples

---

# 5. Data Splitting

A fixed **70/15/15 split** is generated using seed 42.

| Split      | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Test       |        15% |

The same split is reused across all methods.

### Experimental integrity

The test set is isolated.

Test samples are never used for:

* Training
* Replay-memory construction
* Hyperparameter tuning
* Early stopping
* Model selection
* Replay selection
* Debugging decisions based on test performance

Replay memory contains only training samples from previously completed tasks.

---

# 6. Model

The project uses a convolutional neural network implemented in **PyTorch**.

The model receives:

```text
[B, 2, 128]
```

and produces:

```text
[B, 11]
```

classification logits.

### Architecture

```text
Input
  │
  │ [2 × 128] IQ signal
  ↓
Conv1D
64 channels
  ↓
Conv1D
64 channels
  ↓
Conv1D
128 channels
  ↓
Pooling
  ↓
Fully Connected
128 hidden units
  ↓
Dropout
0.3
  ↓
11-class logits
```

### Model configuration

| Parameter         |       Value |
| ----------------- | ----------: |
| Input channels    |           2 |
| Conv channels     | 64, 64, 128 |
| Kernel size       |           3 |
| Pool size         |           2 |
| FC hidden units   |         128 |
| Dropout           |         0.3 |
| Number of classes |          11 |
| Parameters        |      55,691 |

The model architecture is kept unchanged across the continual-learning comparisons.

The model does not directly receive the task ID or SNR as an input feature.

---

# 7. Methods Compared

RF-CL evaluates four experimental settings.

## 7.1 Naive Sequential Learning

The model learns each task sequentially.

At task `t`:

```text
Current task training data
          ↓
       Model
          ↓
       Update
```

No previous samples are replayed.

This provides the primary catastrophic-forgetting baseline.

---

## 7.2 Random Replay

Random Replay maintains a fixed memory of samples from previously completed tasks.

Training uses:

```text
Current Task Data
       +
Previous Replay Memory
       ↓
     Training
```

The replay memory capacity is:

```text
2,000 samples
```

Samples are selected randomly from the eligible historical training data.

---

## 7.3 SNR-Aware Replay

SNR-Aware Replay is the primary continual-learning method.

Instead of treating all historical samples identically, memory construction considers:

```text
Task
  +
SNR
  +
Class
```

The selection process aims to preserve diversity across the historical domains while operating under the same fixed memory budget.

Conceptually:

```text
Previous Memory + Current Task Training Data
                    │
                    ↓
          Group by task / SNR / class
                    │
                    ↓
          Balanced selection
                    │
                    ↓
             2,000 samples
```

The method is intentionally simple and reproducible.

The project does **not** claim replay itself as novel. The contribution is the application of an **SNR-aware replay strategy to RF domain-incremental learning under evolving SNR conditions**.

---

## 7.4 Joint Training

Joint Training provides an offline reference condition.

Instead of learning tasks sequentially, the model receives the training data from all five tasks together:

```text
T1 ─┐
T2 ─┤
T3 ─┼──→ Combined training set ──→ Model
T4 ─┤
T5 ─┘
```

The joint experiment uses all:

```text
154,000 training samples
```

This setting does not represent continual learning because future task training data is available simultaneously.

It is therefore used as a reference/upper-bound-style comparison rather than as a continual-learning method.

---

# 8. Replay Protocol

For replay methods, the training sequence follows:

```text
Task t arrives
      ↓
Current Task Training Data
      +
Memory from Tasks 1...(t-1)
      ↓
Train Model
      ↓
Evaluate on all seen test tasks
      ↓
Update Replay Memory
      ↓
Move to Task t+1
```

A key experimental constraint is that the newly updated memory is **not used to train the same task again**.

This prevents accidental reuse of post-training memory in the current training stage.

---

# 9. Evaluation Protocol

After completing each task, the model is evaluated on the test sets of **all tasks seen so far**.

The accuracy matrix therefore has the structure:

```text
             Test T1  Test T2  Test T3  Test T4  Test T5
After T1        A11
After T2        A21      A22
After T3        A31      A32      A33
After T4        A41      A42      A43      A44
After T5        A51      A52      A53      A54      A55
```

Future test tasks are not evaluated as part of the continual-learning matrix.

---

# 10. Metrics

## Average Accuracy

Final Average Accuracy is calculated after T5:

```text
AA = (A51 + A52 + A53 + A54 + A55) / 5
```

Higher is better.

---

## Average Forgetting

For tasks T1 through T4:

```text
F1 = A11 - A51
F2 = A22 - A52
F3 = A33 - A53
F4 = A44 - A54
```

Average Forgetting:

```text
AF = (F1 + F2 + F3 + F4) / 4
```

Lower is better.

T5 is excluded because there is no later task after T5 from which to measure forgetting.

---

## Additional Metrics

The evaluation framework also computes:

* Overall accuracy
* Macro F1
* Per-class accuracy
* Confusion matrix
* Number of evaluated samples
* Average Accuracy
* Average Forgetting

The confusion matrix contains all:

```text
11 × 11
```

modulation classes.

---

# 11. Experimental Configuration

The main experiments use:

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

training:
  batch_size: 128
  epochs: 10
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 3

replay:
  memory_size: 2000
  methods:
    - random
    - snr_aware
```

Experiments are run on CPU unless another device is explicitly configured.

---

# 12. Final Experimental Results

The following results come from the completed 10-epoch experiments on the real RadioML 2016.10a dataset.

## Final Accuracy After T5

| Method               |   Test T1 |   Test T2 |   Test T3 | Test T4 |   Test T5 | Average Accuracy |
| -------------------- | --------: | --------: | --------: | ------: | --------: | ---------------: |
| **Naive Sequential** |     14.1% |     13.9% |     13.9% |   19.9% |     14.5% |        **15.3%** |
| **Joint Training**   |     82.7% |     82.1% |     73.6% |   30.7% |      9.8% |        **55.8%** |
| **Random Replay**    |     63.3% |     62.8% |     48.8% |   27.9% |     14.2% |        **43.4%** |
| **SNR-Aware Replay** | **74.9%** | **75.0%** | **58.8%** |   23.4% | **14.4%** |        **49.3%** |

### Key observation

After the model has learned all five domains, SNR-Aware Replay retains:

* **74.9%** on T1
* **75.0%** on T2
* **58.8%** on T3

This demonstrates substantially stronger retention of earlier domains than the naive sequential baseline.

---

# 13. Continual-Learning Results

## Random Replay

Final Average Accuracy:

```text
43.4%
```

Average Forgetting:

```text
19.6%
```

Random Replay substantially improves retention compared with naive sequential learning.

---

## SNR-Aware Replay

Final Average Accuracy:

```text
49.3%
```

Average Forgetting:

```text
12.9%
```

Compared with Random Replay:

```text
Average Accuracy:
43.4% → 49.3%
Improvement ≈ +5.9 percentage points
```

```text
Average Forgetting:
19.6% → 12.9%
Reduction ≈ 6.7 percentage points
```

The two replay methods use the same:

* Dataset
* Model architecture
* Memory capacity
* Training protocol
* Optimizer
* Learning rate
* Batch size
* Epoch configuration
* Random seed
* Fixed data splits

The principal difference is the memory-selection strategy.

---

# 14. Naive Sequential Accuracy Matrix

```text
             Test T1  Test T2  Test T3  Test T4  Test T5
After T1       0.821
After T2       0.813    0.867
After T3       0.754    0.796    0.771
After T4       0.314    0.321    0.348    0.359
After T5       0.141    0.139    0.139    0.199    0.145
```

The sharp decline from the diagonal values learned earlier in the sequence to the final T5 values demonstrates catastrophic forgetting.

---

# 15. Random Replay Accuracy Matrix

```text
             Test T1  Test T2  Test T3  Test T4  Test T5
After T1       0.840
After T2       0.837    0.843
After T3       0.824    0.824    0.769
After T4       0.762    0.745    0.569    0.360
After T5       0.633    0.628    0.488    0.279    0.142
```

---

# 16. SNR-Aware Replay Accuracy Matrix

```text
             Test T1  Test T2  Test T3  Test T4  Test T5
After T1       0.840
After T2       0.851    0.857
After T3       0.819    0.826    0.779
After T4       0.775    0.766    0.548    0.359
After T5       0.749    0.750    0.588    0.234    0.144
```

The final matrix highlights the improved retention of the earlier high- and medium-SNR domains.

---

# 17. Joint Training Reference

Joint Training achieved:

| Task     | Test Accuracy |
| -------- | ------------: |
| T1       |         82.7% |
| T2       |         82.1% |
| T3       |         73.6% |
| T4       |         30.7% |
| T5       |          9.8% |
| **Mean** |     **55.8%** |

Joint Training is not a continual-learning method because it has access to all five task training sets simultaneously.

Its purpose is to provide a reference for understanding the performance gap between sequential learning and unrestricted joint access to the training domains.

---

# 18. Main Findings

The experiments demonstrate three important observations.

### 1. Sequential learning suffers from catastrophic forgetting

The Naive Sequential model starts with strong performance on early tasks but loses most of that performance after training on later SNR domains.

For example:

```text
T1 after learning T1: 82.1%
T1 after learning T5: 14.1%
```

---

### 2. Replay substantially improves retention

Random Replay maintains a much larger fraction of previously learned knowledge:

```text
Naive T1 after T5:   14.1%
Random Replay:       63.3%
```

---

### 3. SNR-aware selection improves replay efficiency

Under the same 2,000-sample memory budget:

```text
Random Replay AA:       43.4%
SNR-Aware Replay AA:    49.3%
```

and:

```text
Random Replay AF:       19.6%
SNR-Aware Replay AF:    12.9%
```

This suggests that preserving diversity across the evolving SNR domains can improve continual-learning retention compared with purely random replay.

---

# 19. Project Structure

```text
rf-continual-learning/
│
├── data/
│   ├── raw/
│   │   └── RML2016.10a_dict.pkl
│   └── splits/
│       └── splits_seed42.json
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── prepare_data.py
│   │   ├── raw_loader.py
│   │   ├── splits.py
│   │   ├── synthetic.py
│   │   └── tasks.py
│   │
│   ├── models/
│   │   └── cnn.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── seed.py
│   │   └── synthetic.py
│   │
│   ├── continual/
│   │   ├── naive.py
│   │   ├── joint.py
│   │   ├── memory.py
│   │   └── replay.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── evaluate.py
│       └── plots.py
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_replay.py
│   └── run_ablation.py
│
├── tests/
│   ├── test_continual.py
│   ├── test_metrics.py
│   └── test_model.py
│
├── frontend/
│   ├── src/
│   ├── index.html
│   └── package.json
│
├── config.yaml
├── README.md
└── .gitignore
```

---

# 20. Running the Experiments

## Baselines

Run Naive Sequential and Joint Training:

```bash
python -m experiments.run_baselines \
    --data real \
    --output-dir results/baselines_final \
    --verbose
```

---

## Replay Methods

Run Random Replay and SNR-Aware Replay:

```bash
python -m experiments.run_replay \
    --output-dir results/replay_final \
    --verbose
```

---

## Ablation

Run the replay comparison:

```bash
python -m experiments.run_ablation \
    --output-dir results/ablation_final \
    --verbose
```

---

# 21. Running Tests

Run the complete test suite:

```bash
python -m pytest tests -v
```

The tests cover:

* Replay memory capacity and retrieval
* Replay-memory validation
* Deterministic replay selection
* SNR-aware grouping and selection
* Training-batch construction
* Accuracy calculation
* Macro F1
* Per-class accuracy
* Confusion matrices
* Average Accuracy
* Average Forgetting
* Five-task accuracy matrices
* Incomplete matrix validation
* Zero-division handling
* Model behavior

Synthetic data is used only for software-level unit tests and is **not used as a reported research result**.

---

# 22. Results and Figures

Experiment outputs are stored under:

```text
results/
```

Typical outputs include:

```text
results/
├── baselines_final/
├── replay_final/
├── ablation_final/
└── figures/
```

Generated results include:

* Accuracy matrices
* Per-task metrics
* Macro F1
* Per-class accuracy
* Confusion matrices
* Average Accuracy
* Average Forgetting
* Method comparisons

Generated experiment outputs and the raw dataset are excluded from version control where appropriate.

---

# 23. Frontend Dashboard

The project includes a research-oriented frontend for demonstrating the experimental results.

The dashboard provides views for:

### Overview

* Key project metrics
* Method comparison
* Accuracy matrix
* Final task accuracy
* Average Accuracy
* Average Forgetting

### Task Progress

* T1 → T5 task sequence
* SNR ranges
* Fixed experimental protocol
* Dataset split information

### Replay Memory

* Current task data
* Historical replay memory
* Training data composition
* Updated memory
* Random vs SNR-aware replay strategy

### Evaluation

* Accuracy
* Macro F1
* Per-class accuracy
* Confusion matrix
* Average Accuracy
* Average Forgetting

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

---

# 24. Reproducibility

The experiments use a fixed random seed:

```text
42
```

Reproducibility is supported through:

* Fixed task definitions
* Fixed train/validation/test splits
* Fixed random seed
* Identical model architecture
* Identical training configuration
* Deterministic replay selection where applicable
* Isolated test sets
* Explicit replay-memory capacity
* Automated unit tests

---

# 25. Experimental Fairness

The comparison between Random Replay and SNR-Aware Replay is designed to isolate the effect of memory selection.

Both methods use:

```text
Same dataset
Same task sequence
Same splits
Same model
Same initialization
Same seed
Same optimizer
Same learning rate
Same batch size
Same epochs
Same memory capacity
Same evaluation protocol
```

Only the replay-memory selection strategy differs.

This allows the observed difference to be attributed more directly to the replay-selection mechanism rather than changes in the surrounding training setup.

---

# 26. Limitations

The current study has several limitations.

1. The experiments use a single RadioML 2016.10a dataset.
2. The task sequence is defined using SNR domains only.
3. The replay memory is fixed at 2,000 samples.
4. The primary experiments use a single random seed.
5. The CNN architecture is relatively compact.
6. SNR-aware replay is a simple diversity-preserving strategy rather than a learned memory-management algorithm.
7. The current study focuses on classification performance and forgetting rather than deployment on physical RF hardware.

These limitations provide opportunities for future investigation.

---

# 27. Future Work

Potential extensions include:

* Testing multiple random seeds
* Comparing different replay-memory sizes
* Testing alternative memory-selection strategies
* Adaptive memory allocation based on SNR
* More sophisticated domain-aware sampling
* Streaming/online evaluation
* Real-world RF datasets
* Hardware-based RF signal acquisition
* Lightweight deployment on edge devices
* Comparison with additional continual-learning algorithms
* Statistical significance analysis across repeated runs

---

# 28. Conclusion

RF-CL investigates continual learning for RF modulation classification under evolving SNR conditions.

The experimental sequence demonstrates the effect of changing RF domains on a fixed 11-class classification problem.

The main progression is:

```text
Naive Sequential
       ↓
Severe catastrophic forgetting
       ↓
Random Replay
       ↓
Improved knowledge retention
       ↓
SNR-Aware Replay
       ↓
Further improvement under the same memory budget
```

The completed experiments show:

```text
Random Replay
Average Accuracy = 43.4%
Average Forgetting = 19.6%

SNR-Aware Replay
Average Accuracy = 49.3%
Average Forgetting = 12.9%
```

Therefore, within this experimental setup, **SNR-aware replay provides better continual-learning retention than random replay while using the same fixed 2,000-sample memory budget and the same underlying model and training protocol.**

---

# 29. Team Contributions

The project is divided into modular responsibilities.

### Data Pipeline

Responsible for:

* RadioML 2016.10a loading
* Task construction
* Fixed train/validation/test splits
* Dataset validation

### Model & Training

Responsible for:

* CNN baseline
* Training infrastructure
* Naive Sequential baseline
* Joint Training reference

### Continual Learning & Evaluation

Responsible for:

* Replay memory
* Random Replay
* SNR-Aware Replay
* Continual-learning evaluation
* Accuracy matrices
* Average Accuracy
* Average Forgetting
* Per-class metrics
* Confusion matrices
* Ablation experiments
* Continual-learning tests
* Result visualization

---

# 30. Key Takeaway

> **RF-CL shows that when RF modulation classification is exposed to evolving SNR domains sequentially, naive learning can suffer severe catastrophic forgetting. A fixed replay memory substantially improves retention, and an SNR-aware selection strategy further improves the trade-off between learning new domains and retaining previously learned knowledge.**

**RF-CL · Continual Learning for RF Modulation Classification Under Evolving SNR Conditions**
