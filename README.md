# CompactionRL-style GRPO on HotpotQA

This repository contains a lightweight single-GPU experiment inspired by **CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents**.

The goal is to test whether a small language model can learn to generate a useful compact memory using only a downstream task reward.

The experiment uses:

* **Model:** Qwen3-0.6B
* **Dataset:** HotpotQA Distractor
* **Training samples:** 1,000
* **Evaluation samples:** 200
* **RL algorithm:** GRPO
* **Fine-tuning:** LoRA
* **Reward:** HotpotQA answer F1
* **Hardware:** Single GPU
* **Framework:** Hugging Face TRL

> This is a lightweight single-GPU approximation of CompactionRL, not an exact reproduction of the original PPO training pipeline.

---

## 1. Motivation

Long-horizon agents eventually exceed their context window as tool observations and intermediate states accumulate.

A simple solution is to summarize previous context and continue from the compressed state.

However, a separately trained summarizer may optimize generic summary quality rather than information that is actually useful for completing the task.

CompactionRL instead trains context compaction using the final task reward.

The core idea is:

```text
Long Context
     |
     v
Compact Memory
     |
     v
Reason / Answer
     |
     v
Final Task Reward
```

There is no gold summary.

The model must learn which information should be retained because useful memories lead to better final answers.

---

## 2. Experiment

For each HotpotQA example, the model receives:

```text
Question
+
Multiple evidence documents
```

and generates:

```text
<MEMORY>
compact useful information
</MEMORY>

<ANSWER>
final answer
</ANSWER>
```

The `<MEMORY>` section does not receive a separate supervision signal.

Instead, the final answer is compared with the HotpotQA gold answer and an F1 score is used as the reinforcement-learning reward.

Conceptually:

```text
Question + Documents
        |
        v
     Qwen3
        |
        +--------------------+
        |                    |
        v                    v
    <MEMORY>             <ANSWER>
        |                    |
        +---------+----------+
                  |
                  v
             Answer F1
                  |
                  v
             GRPO Update
```

The same terminal reward therefore influences the entire generated sequence, including the memory tokens.

---

## 3. Results

Evaluation was performed on 200 HotpotQA validation examples.

| Model               |         EM |         F1 |
| ------------------- | ---------: | ---------: |
| Before RL training  |     0.1000 |     0.1585 |
| After GRPO training | **0.1400** | **0.2182** |

Absolute improvement:

```text
EM: 0.1000 -> 0.1400
    +0.0400

F1: 0.1585 -> 0.2182
    +0.0597
```

Relative improvement:

```text
EM: +40.0%
F1: +37.7%
```

The experiment therefore shows a measurable improvement after reinforcement learning despite using only 1,000 training examples.

The result suggests that task-level reward can provide a useful learning signal for jointly improving compact memory generation and downstream answering.

Because this is a small experiment and only a single run, the result should not be interpreted as statistically conclusive.

---

## 4. Difference from the Original CompactionRL

The original CompactionRL algorithm uses PPO with a critic and explicitly models multiple execution and compaction segments.

A trajectory is closer to:

```text
Execution segment
      |
      v
Compaction
      |
      v
Context reset
      |
      v
Execution segment
      |
      v
Compaction
      |
      v
Context reset
      |
      v
Final answer
      |
      v
Task reward
```

The original method additionally uses:

* PPO
* learned value function / critic
* token-level advantages
* local GAE
* cross-trajectory advantage correction
* explicit context removal and resume
* multiple compaction events within one trajectory
* the same policy for execution and summarization

This repository instead uses:

```text
Question + Documents
        |
        v
<MEMORY>
        |
        v
<ANSWER>
        |
        v
F1 reward
        |
        v
GRPO
```

Therefore, this implementation should be considered a **CompactionRL-style single-GPU experiment**, rather than an exact reproduction.

---

## 5. Why GRPO?

The original paper uses PPO.

Running the full architecture efficiently requires separate resources for:

```text
Actor
Critic
Rollout generation
```

which is inconvenient on a single consumer GPU.

For this experiment, GRPO is used instead because it removes the learned critic.

For each prompt, multiple candidate responses are generated:

```text
Prompt
   |
   +--> Completion 1 --> Reward 1
   |
   +--> Completion 2 --> Reward 2
             |
             v
       Relative Advantage
             |
             v
          GRPO
```

This substantially simplifies single-GPU experimentation.

The trade-off is that the optimization algorithm is no longer identical to the original CompactionRL method.

---

## 6. Dataset

The experiment uses the **HotpotQA Distractor** configuration.

HotpotQA is suitable for this experiment because answering a question often requires combining information distributed across multiple documents.

Example:

```text
Question:
What nationality was the author of X?

Document A:
X was written by John Doe.

Document B:
John Doe was an Australian novelist.
```

A useful compact memory could be:

```text
<MEMORY>
X was written by John Doe.
John Doe was Australian.
</MEMORY>
```

followed by:

```text
<ANSWER>
Australian
</ANSWER>
```

This makes the task useful for studying whether the model learns to retain downstream-relevant information.

---

## 7. Training Configuration

The main experimental setup is approximately:

```python
MODEL_NAME = "Qwen/Qwen3-0.6B"

TRAIN_N = 1000
EVAL_N = 200

NUM_GENERATIONS = 2

LEARNING_RATE = 2e-6
```

LoRA is applied to attention projections:

```python
LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)
```

A relatively short context and completion length can be used to make training practical on one GPU.

For example:

```python
MAX_DOCUMENT_TOKENS = 600
MAX_COMPLETION_LENGTH = 96
```

---

## 8. Reward

The main reward is HotpotQA token-level F1.

For prediction \(P\) and ground-truth answer \(G\):

$$
Precision =
\frac{|P \cap G|}{|P|}
$$

$$
Recall =
\frac{|P \cap G|}{|G|}
$$

$$
F1 =
\frac{2PR}{P+R}
$$

The training reward is approximately:

$$
R = F1(\text{prediction},\text{gold})
$$

with an optional very small formatting bonus for producing the expected `<MEMORY>` and `<ANSWER>` structure.

Importantly, there is no reward such as:

```text
summary similarity
ROUGE(summary, gold_summary)
summary BLEU
```

because no gold summary is provided.

The desired behavior must emerge from the downstream QA reward.

---

## 9. Installation

Install the required libraries:

```bash
pip install -U \
    torch \
    transformers \
    datasets \
    peft \
    accelerate \
    trl
```

Optional generation acceleration can be enabled with a compatible vLLM installation.

---

## 10. Run

Run training with:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_compaction_grpo.py
```

The script:

```text
1. Downloads HotpotQA
2. Selects 1,000 training examples
3. Selects 200 validation examples
4. Builds compact-memory prompts
5. Loads Qwen3-0.6B
6. Applies LoRA
7. Performs GRPO training
8. Saves the trained adapter
9. Evaluates on the validation set
10. Reports EM and F1
```

---

## 11. Output

The trained model and evaluation results are saved under:

```text
qwen3_compaction_grpo/
```

Example:

```text
qwen3_compaction_grpo/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── eval_results.json
```

`eval_results.json` contains the aggregate metrics and individual predictions.

Example:

```json
{
  "metrics": {
    "EM": 0.14,
    "F1": 0.2182,
    "num_examples": 200
  }
}
```

---

## 12. Evaluation Metrics

### Exact Match

Exact Match is 1 only when the normalized prediction exactly matches the normalized gold answer.

$$
EM =
\mathbb{1}
[
normalize(P)=normalize(G)
]
$$

### F1

F1 gives partial credit based on overlapping answer tokens.

This is particularly useful when the predicted answer is semantically close but not an exact string match.

---

## 13. Observed Improvement

The most important result from the initial experiment is:

```text
Before training
EM = 0.1000
F1 = 0.1585

After training
EM = 0.1400
F1 = 0.2182
```

The increase in both metrics indicates that the RL update is learning a useful policy rather than merely changing output formatting.

In particular, F1 improves by approximately 5.97 percentage points.

```text
0.2182 - 0.1585 = 0.0597
```

The next question is whether the improvement specifically comes from learning better compact memories.

That requires additional ablation experiments.

---

## 14. Recommended Ablations

The most useful follow-up experiment is to compare three settings.

### No Memory

```text
Question + Documents
        |
        v
      Answer
```

This tests ordinary GRPO QA training.

### Memory, but Memory Not Trained

```text
Question + Documents
        |
        v
     Memory
        |
       stop-gradient
        |
        v
      Answer
```

This approximates the CompactionRL ablation where summaries are generated but excluded from policy optimization.

### Memory + Joint RL Training

```text
Question + Documents
        |
        v
     Memory
        |
        v
      Answer
        |
        v
 Final reward updates
 Memory + Answer
```

This is the current experiment.

The comparison:

```text
No-memory GRPO
vs.
Frozen/untrained memory
vs.
Jointly trained memory
```

would provide much stronger evidence that the gain is actually caused by reinforcement-learned compaction.

---

## 15. Limitations

This experiment has several important limitations.

First, only 1,000 training examples are used.

Second, the experiment currently reports a single training run rather than results over multiple random seeds.

Third, HotpotQA is much shorter than the software-engineering trajectories used by the original CompactionRL paper.

Fourth, the generated memory does not actually replace an overflowing context window. The memory and answer are generated within one response.

Finally, GRPO is used instead of the original PPO + critic architecture.

Therefore, the experiment demonstrates the basic learning principle but not the complete long-horizon CompactionRL system.

---

## 16. Next Steps

A stronger reproduction would progressively introduce:

```text
1. No-memory / memory ablation
2. Multiple random seeds
3. Explicit context truncation
4. Multiple compaction points
5. Resume from compacted memory
6. PPO critic
7. Token-level GAE
8. Cross-trajectory advantage correction
9. Agent/tool interaction environment
10. SWE-Dev or SWE-bench-style trajectories
```

The final architecture would more closely resemble:

```text
Environment
    |
    v
Execution
    |
context nearly full
    |
    v
Summary / Compaction
    |
discard old context
    |
    v
Resume from summary
    |
    v
Execution
    |
    ...
    |
    v
Final task reward
    |
    +----------------------+
    |                      |
    v                      v
execution tokens       summary tokens
    |                      |
    +----------PPO---------+
```

---

## Summary

This experiment provides a small-scale demonstration of reinforcement-learned context compaction on a single GPU.

Using Qwen3-0.6B and only 1,000 HotpotQA training examples, GRPO training improved validation performance from:

```text
EM: 0.1000 -> 0.1400
F1: 0.1585 -> 0.2182
```

The result supports further investigation into learning compact memories directly from downstream task rewards rather than training summarization independently.

## 2-Step
```
Before: {'N': 100, 'EM': 0.17, 'F1': 0.26895141969376907}
Both : {'N': 100, 'EM': 0.18, 'F1': 0.283643802434125}
F1 Only : {'N': 100, 'EM': 0.18, 'F1': 0.2672583250646131}
Summary Only : {'N': 100, 'EM': 0.17, 'F1': 0.2713038802156449}
```
