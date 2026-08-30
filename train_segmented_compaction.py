# train_segmented_compaction.py

import os
import re
import gc
import json
import math
import random
import string
import collections
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen3-0.6B"

# Previous experiment adapter
PREVIOUS_ADAPTER = "./qwen3_compaction_grpo"

OUTPUT_DIR = "./qwen3_segmented_compaction"

SEED = 42

# EXACT SAME DATA SPLIT AS PREVIOUS CODE
TRAIN_N = 1000
EVAL_N = 200

# ------------------------------------------------------------
# Compaction
# ------------------------------------------------------------

# Documents are processed incrementally.
DOCS_PER_SEGMENT = 3

# Summary length
MAX_SUMMARY_TOKENS = 64

# Final answer length
MAX_ANSWER_TOKENS = 32

# Keep compact memory short
MAX_MEMORY_INPUT_TOKENS = 384

# ------------------------------------------------------------
# RL
# ------------------------------------------------------------

# Two trajectories / same question
# => group-relative advantage
NUM_ROLLOUTS = 2

LR = 2e-6

GRAD_ACCUM = 4

# train 1 epoch over same 1000 examples
EPOCHS = 1

TEMPERATURE = 0.8
TOP_P = 0.95

# generation
DO_SAMPLE = True

# KL-ish clipping for policy ratio
PPO_CLIP = 0.2

# Prevent very long trajectories
MAX_COMPACTIONS = 3

# Evaluation
EVAL_BEFORE = True
EVAL_AFTER = True

DEVICE = "cuda"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# Metrics
# ============================================================

def normalize_answer(s):

    if s is None:
        return ""

    s = str(s)

    def remove_articles(text):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def white_space_fix(text):
        return " ".join(
            text.split()
        )

    def remove_punc(text):

        exclude = set(
            string.punctuation
        )

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    return white_space_fix(
        remove_articles(
            remove_punc(
                s.lower()
            )
        )
    )


def exact_match(pred, gold):

    return float(
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


def answer_f1(pred, gold):

    pred = normalize_answer(pred)
    gold = normalize_answer(gold)

    special = {
        "yes",
        "no",
        "noanswer",
    }

    if pred in special or gold in special:
        return float(pred == gold)

    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if not pred_tokens or not gold_tokens:
        return float(
            pred_tokens == gold_tokens
        )

    common = (
        collections.Counter(pred_tokens)
        &
        collections.Counter(gold_tokens)
    )

    num_same = sum(
        common.values()
    )

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        / len(pred_tokens)
    )

    recall = (
        num_same
        / len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# ============================================================
# Dataset
# ============================================================

def load_same_data():

    print("Loading HotpotQA...")

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # EXACT SAME shuffling logic as previous experiment.
    # --------------------------------------------------------

    train = (
        ds["train"]
        .shuffle(seed=SEED)
        .select(range(TRAIN_N))
    )

    eval_ds = (
        ds["validation"]
        .shuffle(seed=SEED)
        .select(range(EVAL_N))
    )

    print("Train:", len(train))
    print("Eval :", len(eval_ds))

    return train, eval_ds


def get_documents(example):

    docs = []

    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    for title, sent_list in zip(
        titles,
        sentences,
    ):

        docs.append(
            f"[{title}]\n"
            + " ".join(sent_list)
        )

    return docs


# ============================================================
# Prompts
# ============================================================

SUMMARY_SYSTEM = """
You are the memory component of a long-horizon agent.

Compress the current state so that another instance of yourself can
continue solving the original question after the old context is deleted.

Preserve:
- facts needed for answering the question
- entity relationships
- intermediate deductions
- useful names, dates, locations and numbers
- unresolved information

Remove irrelevant details.

Return ONLY the compact memory.
""".strip()


ANSWER_SYSTEM = """
Answer the multi-hop question using the compact memory.

Return ONLY the short final answer.
Do not explain.
""".strip()


def build_summary_messages(
    question,
    old_memory,
    new_docs,
):

    docs_text = "\n\n".join(
        new_docs
    )

    if old_memory:

        state = f"""
Original question:
{question}

Previous compact memory:
{old_memory}

New observations:
{docs_text}

Update the compact memory.
""".strip()

    else:

        state = f"""
Original question:
{question}

Observations:
{docs_text}

Create compact memory for continuing the task.
""".strip()

    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM,
        },
        {
            "role": "user",
            "content": state,
        },
    ]


def build_answer_messages(
    question,
    memory,
):

    content = f"""
Question:
{question}

Compact memory:
{memory}

Answer the original question.
""".strip()

    return [
        {
            "role": "system",
            "content": ANSWER_SYSTEM,
        },
        {
            "role": "user",
            "content": content,
        },
    ]


# ============================================================
# Trajectory objects
# ============================================================

@dataclass
class Segment:

    prompt_ids: torch.Tensor

    generated_ids: torch.Tensor

    old_logprobs: torch.Tensor

    kind: str


@dataclass
class Trajectory:

    segments: List[Segment]

    reward: float

    answer: str

    memory: str


# ============================================================
# Chat templating
# ============================================================

def chat_to_ids(
    tokenizer,
    messages,
):

    try:

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    return ids


# ============================================================
# Generate one segment
# ============================================================

@torch.inference_mode()
def generate_segment(
    model,
    tokenizer,
    messages,
    max_new_tokens,
    do_sample=True,
):

    prompt_ids = chat_to_ids(
        tokenizer,
        messages,
    ).to(DEVICE)

    input_ids = (
        prompt_ids.unsqueeze(0)
    )

    outputs = model.generate(
        input_ids=input_ids,

        max_new_tokens=max_new_tokens,

        do_sample=do_sample,

        temperature=(
            TEMPERATURE
            if do_sample
            else None
        ),

        top_p=(
            TOP_P
            if do_sample
            else None
        ),

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),

        use_cache=True,
    )

    generated = outputs[
        0,
        prompt_ids.shape[0]:
    ]

    return (
        prompt_ids.detach().cpu(),
        generated.detach().cpu(),
    )


# ============================================================
# Log probabilities
# ============================================================

def sequence_logprobs(
    model,
    prompt_ids,
    generated_ids,
    requires_grad=True,
):

    """
    Returns log p(generated token | prompt + previous generated tokens).

    Shape:
        [generated_length]
    """

    prompt_ids = prompt_ids.to(
        DEVICE
    )

    generated_ids = generated_ids.to(
        DEVICE
    )

    full_ids = torch.cat(
        [
            prompt_ids,
            generated_ids,
        ],
        dim=0,
    )

    x = full_ids[:-1].unsqueeze(0)
    target = full_ids[1:]

    if requires_grad:

        outputs = model(
            input_ids=x,
            use_cache=False,
        )

    else:

        with torch.no_grad():

            outputs = model(
                input_ids=x,
                use_cache=False,
            )

    logits = outputs.logits[0]

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    token_log_probs = (
        log_probs
        .gather(
            -1,
            target.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    # Prediction corresponding to the first generated token.
    start = (
        prompt_ids.shape[0] - 1
    )

    generated_log_probs = (
        token_log_probs[
            start:
            start + generated_ids.shape[0]
        ]
    )

    return generated_log_probs


# ============================================================
# Memory truncation
# ============================================================

def truncate_memory(
    tokenizer,
    text,
):

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if (
        len(ids)
        <= MAX_MEMORY_INPUT_TOKENS
    ):
        return text

    ids = ids[
        -MAX_MEMORY_INPUT_TOKENS:
    ]

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )


# ============================================================
# One complete Compaction trajectory
# ============================================================

def rollout_one(
    model,
    tokenizer,
    example,
    training=True,
):

    question = example[
        "question"
    ]

    gold = example[
        "answer"
    ]

    docs = get_documents(
        example
    )

    memory = ""

    segments = []

    # --------------------------------------------------------
    # Split observations into chunks
    # --------------------------------------------------------

    chunks = [
        docs[
            i:
            i + DOCS_PER_SEGMENT
        ]

        for i in range(
            0,
            len(docs),
            DOCS_PER_SEGMENT,
        )
    ]

    # We don't want unlimited compaction.
    chunks = chunks[
        :MAX_COMPACTIONS
    ]

    # --------------------------------------------------------
    # SUMMARY / COMPACTION
    # --------------------------------------------------------

    for chunk in chunks:

        messages = (
            build_summary_messages(
                question,
                memory,
                chunk,
            )
        )

        (
            prompt_ids,
            generated_ids,
        ) = generate_segment(
            model,
            tokenizer,
            messages,
            MAX_SUMMARY_TOKENS,
            do_sample=training,
        )

        memory = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        memory = truncate_memory(
            tokenizer,
            memory,
        )

        if training:

            old_lp = (
                sequence_logprobs(
                    model,
                    prompt_ids,
                    generated_ids,
                    requires_grad=False,
                )
                .detach()
                .cpu()
            )

            segments.append(
                Segment(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    old_logprobs=old_lp,
                    kind="summary",
                )
            )

        # ----------------------------------------------------
        # IMPORTANT COMPACTION STEP
        #
        # Old documents are NOT preserved here.
        #
        # The next model invocation receives:
        #
        #     memory + new chunk
        #
        # instead of the complete previous history.
        #
        # ----------------------------------------------------

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    answer_messages = (
        build_answer_messages(
            question,
            memory,
        )
    )

    (
        prompt_ids,
        generated_ids,
    ) = generate_segment(
        model,
        tokenizer,
        answer_messages,
        MAX_ANSWER_TOKENS,
        do_sample=training,
    )

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    # Keep only first useful line
    answer = (
        answer.split("\n")[0]
        .strip()
    )

    reward = answer_f1(
        answer,
        gold,
    )

    if training:

        old_lp = sequence_logprobs(
            model,
            prompt_ids,
            generated_ids,
            requires_grad=False,
        ).detach().cpu()

        segments.append(
            Segment(
                prompt_ids=prompt_ids,
                generated_ids=generated_ids,
                old_logprobs=old_lp,
                kind="answer",
            )
        )

    return Trajectory(
        segments=segments,
        reward=reward,
        answer=answer,
        memory=memory,
    )


# ============================================================
# PPO-like loss over segmented trajectory
# ============================================================

def trajectory_loss(
    model,
    trajectory,
    advantage,
):

    """
    Final task reward / advantage is shared by:

        summary 1
        summary 2
        summary 3
        answer

    All generated assistant tokens are optimized.

    We token-normalize across the complete trajectory.
    """

    losses = []

    token_count = 0

    for segment in trajectory.segments:

        new_logprobs = (
            sequence_logprobs(
                model,
                segment.prompt_ids,
                segment.generated_ids,
                requires_grad=True,
            )
        )

        old_logprobs = (
            segment.old_logprobs
            .to(DEVICE)
        )

        # PPO ratio
        ratio = torch.exp(
            new_logprobs
            - old_logprobs
        )

        adv = torch.tensor(
            advantage,
            dtype=torch.float32,
            device=DEVICE,
        )

        unclipped = (
            ratio * adv
        )

        clipped = (
            torch.clamp(
                ratio,
                1.0 - PPO_CLIP,
                1.0 + PPO_CLIP,
            )
            * adv
        )

        token_loss = -torch.minimum(
            unclipped,
            clipped,
        )

        losses.append(
            token_loss.sum()
        )

        token_count += (
            token_loss.numel()
        )

    if token_count == 0:

        return torch.tensor(
            0.0,
            device=DEVICE,
            requires_grad=True,
        )

    # Token-level normalization like CompactionRL paper.
    return (
        torch.stack(losses).sum()
        / token_count
    )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    dataset,
    label,
):

    model.eval()

    ems = []
    f1s = []

    results = []

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    for i, example in enumerate(
        dataset
    ):

        traj = rollout_one(
            model,
            tokenizer,
            example,
            training=False,
        )

        gold = example[
            "answer"
        ]

        em = exact_match(
            traj.answer,
            gold,
        )

        f1 = answer_f1(
            traj.answer,
            gold,
        )

        ems.append(em)
        f1s.append(f1)

        results.append({
            "question":
                example["question"],

            "gold":
                gold,

            "prediction":
                traj.answer,

            "memory":
                traj.memory,

            "em":
                em,

            "f1":
                f1,
        })

        if (
            (i + 1) % 20 == 0
            or i == 0
        ):

            print(
                f"[{i+1}/{len(dataset)}] "
                f"EM="
                f"{sum(ems)/len(ems):.4f} "
                f"F1="
                f"{sum(f1s)/len(f1s):.4f}"
            )

    metrics = {
        "EM":
            sum(ems) / len(ems),

        "F1":
            sum(f1s) / len(f1s),

        "N":
            len(dataset),
    }

    print(
        "\n",
        metrics,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        OUTPUT_DIR,
        label.replace(
            " ",
            "_"
        ).lower()
        + ".json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "metrics":
                    metrics,

                "examples":
                    results,
            },

            f,

            indent=2,
            ensure_ascii=False,
        )

    return metrics


# ============================================================
# Model
# ============================================================

def load_model():

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    print(
        "Loading base model..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,

            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            ),

            device_map={
                "": 0
            },

            trust_remote_code=True,
        )
    )

    # --------------------------------------------------------
    # Start from previous trained adapter
    # --------------------------------------------------------

    adapter_file = os.path.join(
        PREVIOUS_ADAPTER,
        "adapter_config.json",
    )

    if os.path.exists(
        adapter_file
    ):

        print(
            "Loading previous trained adapter:",
            PREVIOUS_ADAPTER,
        )

        model = PeftModel.from_pretrained(
            model,
            PREVIOUS_ADAPTER,
            is_trainable=True,
        )

    else:

        print(
            "Previous adapter not found."
        )

        print(
            "Creating fresh LoRA."
        )

        config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,

            bias="none",

            task_type="CAUSAL_LM",

            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )

        model = get_peft_model(
            model,
            config,
        )

    model.config.use_cache = True

    model.print_trainable_parameters()

    return (
        model,
        tokenizer,
    )


# ============================================================
# Training
# ============================================================

def train(
    model,
    tokenizer,
    train_dataset,
):

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],

        lr=LR,

        betas=(
            0.9,
            0.99,
        ),

        weight_decay=0.01,
    )

    model.train()

    global_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(EPOCHS):

        order = list(
            range(
                len(train_dataset)
            )
        )

        # Deterministic training order
        rnd = random.Random(
            SEED + epoch
        )

        rnd.shuffle(order)

        for step, idx in enumerate(
            order
        ):

            example = train_dataset[
                idx
            ]

            # ------------------------------------------------
            # Generate 2 complete trajectories for same prompt
            # ------------------------------------------------

            model.eval()

            trajectories = []

            for _ in range(
                NUM_ROLLOUTS
            ):

                trajectories.append(
                    rollout_one(
                        model,
                        tokenizer,
                        example,
                        training=True,
                    )
                )

            rewards = torch.tensor(
                [
                    t.reward
                    for t
                    in trajectories
                ],

                dtype=torch.float32,
            )

            # ------------------------------------------------
            # Group-relative advantages
            # ------------------------------------------------

            if (
                rewards.std(
                    unbiased=False
                )
                > 1e-8
            ):

                advantages = (
                    rewards
                    - rewards.mean()
                ) / (
                    rewards.std(
                        unbiased=False
                    )
                    + 1e-6
                )

            else:

                advantages = (
                    rewards
                    - rewards.mean()
                )

            # ------------------------------------------------
            # Update all summary+answer segments
            # ------------------------------------------------

            model.train()

            total_loss = 0.0

            for trajectory, adv in zip(
                trajectories,
                advantages,
            ):

                loss = trajectory_loss(
                    model,
                    trajectory,
                    float(adv),
                )

                loss = (
                    loss
                    / (
                        GRAD_ACCUM
                        * NUM_ROLLOUTS
                    )
                )

                loss.backward()

                total_loss += (
                    float(
                        loss.detach()
                    )
                )

            if (
                (step + 1)
                % GRAD_ACCUM
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p
                        in model.parameters()
                        if p.requires_grad
                    ],
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            if (
                (step + 1) % 10
                == 0
            ):

                print(
                    f"epoch={epoch+1} "
                    f"sample={step+1}/"
                    f"{len(train_dataset)} "
                    f"reward="
                    f"{rewards.mean().item():.4f} "
                    f"loss="
                    f"{total_loss:.4f}"
                )

            del trajectories

            gc.collect()

            if (
                torch.cuda.is_available()
                and (step + 1) % 20 == 0
            ):

                torch.cuda.empty_cache()

    return model


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    train_dataset, eval_dataset = (
        load_same_data()
    )

    model, tokenizer = (
        load_model()
    )

    # --------------------------------------------------------
    # Evaluation BEFORE segmented Compaction training
    #
    # Same exact eval 200 examples.
    #
    # Note:
    # this starts from your previous GRPO adapter.
    # --------------------------------------------------------

    if EVAL_BEFORE:

        before_metrics = evaluate(
            model,
            tokenizer,
            eval_dataset,
            "before_segmented_training",
        )

    else:

        before_metrics = None

    # --------------------------------------------------------
    # Train on SAME train 1000
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEGMENTED COMPACTION TRAINING")
    print("=" * 70)

    train(
        model,
        tokenizer,
        train_dataset,
    )

    print(
        "Saving adapter..."
    )

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Same exact eval set after training
    # --------------------------------------------------------

    if EVAL_AFTER:

        after_metrics = evaluate(
            model,
            tokenizer,
            eval_dataset,
            "after_segmented_training",
        )

    else:

        after_metrics = None

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        "Before segmented training:",
        before_metrics,
    )

    print(
        "After segmented training :",
        after_metrics,
    )


if __name__ == "__main__":
    main()
