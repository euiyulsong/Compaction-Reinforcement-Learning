# train_batched_segmented_compaction.py

import os
import re
import gc
import json
import random
import string
import collections
from dataclasses import dataclass
from typing import List, Optional

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

# ------------------------------------------------------------
# 이전 single-prompt GRPO에서 학습한 LoRA
#
# 있으면 여기서 이어서 학습.
# 없으면 fresh LoRA 생성.
# ------------------------------------------------------------

PREVIOUS_ADAPTER = "./qwen3_compaction_grpo"

OUTPUT_DIR = "./qwen3_batched_segmented_compaction"

SEED = 42


# ============================================================
# DATA
# ============================================================

# 이전 실험과 동일
TRAIN_N = 1000
EVAL_N = 100


# ============================================================
# BATCH
# ============================================================

# 실제 generation을 동시에 20개 수행
TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 20

# 같은 문제당 trajectory 2개
#
# GRPO-style relative advantage를 계산하기 위함.
NUM_ROLLOUTS = 2


# ============================================================
# COMPACTION
# ============================================================

# HotpotQA context document를 몇 개씩 읽을지
DOCS_PER_SEGMENT = 3

# 최대 summary 횟수
MAX_COMPACTIONS = 3

# summary generation 길이
MAX_SUMMARY_TOKENS = 64

# answer generation 길이
MAX_ANSWER_TOKENS = 32

# 이전 memory가 너무 길어지는 것 방지
MAX_MEMORY_INPUT_TOKENS = 384


# ============================================================
# RL
# ============================================================

LEARNING_RATE = 2e-6

EPOCHS = 1

PPO_CLIP = 0.2

TEMPERATURE = 0.8
TOP_P = 0.95

MAX_GRAD_NORM = 1.0


# ============================================================
# EVAL
# ============================================================

EVAL_BEFORE = True
EVAL_AFTER = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.bfloat16
    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    else torch.float16
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# METRIC
# ============================================================

def normalize_answer(s):

    if s is None:
        return ""

    s = str(s).lower()

    def remove_articles(text):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def remove_punc(text):

        exclude = set(
            string.punctuation
        )

        return "".join(
            c
            for c in text
            if c not in exclude
        )

    def white_space_fix(text):
        return " ".join(
            text.split()
        )

    return white_space_fix(
        remove_articles(
            remove_punc(s)
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

    if (
        pred in special
        or gold in special
    ):
        return float(
            pred == gold
        )

    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if (
        len(pred_tokens) == 0
        or len(gold_tokens) == 0
    ):
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
# DATASET
# ============================================================

def load_same_data():

    print("\nLoading HotpotQA...")

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    # --------------------------------------------------------
    # 이전 실험과 동일
    # --------------------------------------------------------

    train_dataset = (
        ds["train"]
        .shuffle(seed=SEED)
        .select(
            range(
                min(
                    TRAIN_N,
                    len(ds["train"]),
                )
            )
        )
    )

    eval_dataset = (
        ds["validation"]
        .shuffle(seed=SEED)
        .select(
            range(
                min(
                    EVAL_N,
                    len(ds["validation"]),
                )
            )
        )
    )

    print(
        f"Train = {len(train_dataset)}"
    )

    print(
        f"Eval  = {len(eval_dataset)}"
    )

    return (
        train_dataset,
        eval_dataset,
    )


def get_documents(example):

    docs = []

    context = example[
        "context"
    ]

    titles = context[
        "title"
    ]

    sentences = context[
        "sentences"
    ]

    for title, sent_list in zip(
        titles,
        sentences,
    ):

        text = (
            f"[{title}]\n"
            + " ".join(sent_list)
        )

        docs.append(text)

    return docs


# ============================================================
# PROMPTS
# ============================================================

SUMMARY_SYSTEM = """
You are the compact memory component of a long-horizon reasoning agent.

The old context will be deleted after you respond.

Write a compact memory that preserves only information useful for
answering the original question later.

Preserve:
- important entities
- relationships
- facts
- dates and numbers
- intermediate conclusions
- unresolved clues

Do not answer the question unless the answer is already clearly known.
Do not add information not supported by the observations.

Return ONLY the compact memory.
""".strip()


ANSWER_SYSTEM = """
Answer the original question using the compact memory.

Return ONLY the short final answer.
Do not explain your reasoning.
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

        content = f"""
Original question:
{question}

Previous compact memory:
{old_memory}

New observations:
{docs_text}

Rewrite the compact memory so another agent can continue after all
previous context is deleted.
""".strip()

    else:

        content = f"""
Original question:
{question}

Observations:
{docs_text}

Create a compact memory so another agent can continue after these
observations are deleted.
""".strip()

    return [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM,
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def build_answer_messages(
    question,
    memory,
):

    content = f"""
Original question:
{question}

Compact memory:
{memory}

Give the final answer.
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
# DATA CLASSES
# ============================================================

@dataclass
class Segment:

    prompt_ids: torch.Tensor

    generated_ids: torch.Tensor

    old_logprobs: Optional[
        torch.Tensor
    ]

    kind: str


@dataclass
class Trajectory:

    segments: List[Segment]

    reward: float

    answer: str

    memory: str

    gold: str

    question: str


# ============================================================
# CHAT TEMPLATE
# ============================================================

def messages_to_ids(
    tokenizer,
    messages,
):

    try:

        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    except TypeError:

        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return ids


# ============================================================
# BATCH GENERATION
# ============================================================

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    messages_batch,
    max_new_tokens,
    do_sample,
):

    """
    True batched generation.

    messages_batch:
        list[list[dict]]

    Returns:
        [
            (prompt_ids_cpu, generated_ids_cpu),
            ...
        ]
    """

    encoded = []

    for messages in messages_batch:

        ids = messages_to_ids(
            tokenizer,
            messages,
        )

        encoded.append(ids)

    # --------------------------------------------------------
    # LEFT padding is needed for batched causal generation.
    # --------------------------------------------------------

    max_prompt_len = max(
        len(x)
        for x in encoded
    )

    input_ids = []

    attention_masks = []

    pad_id = tokenizer.pad_token_id

    for ids in encoded:

        pad_len = (
            max_prompt_len
            - len(ids)
        )

        padded = (
            [pad_id] * pad_len
            + ids
        )

        mask = (
            [0] * pad_len
            + [1] * len(ids)
        )

        input_ids.append(
            padded
        )

        attention_masks.append(
            mask
        )

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.tensor(
        attention_masks,
        dtype=torch.long,
        device=DEVICE,
    )

    generation_kwargs = {
        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "max_new_tokens":
            max_new_tokens,

        "do_sample":
            do_sample,

        "pad_token_id":
            tokenizer.pad_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "use_cache":
            True,
    }

    if do_sample:

        generation_kwargs[
            "temperature"
        ] = TEMPERATURE

        generation_kwargs[
            "top_p"
        ] = TOP_P

    output = model.generate(
        **generation_kwargs
    )

    results = []

    for i in range(
        len(encoded)
    ):

        # output prefix includes padded input
        generated = output[
            i,
            max_prompt_len:
        ]

        # Stop after first EOS, if present.
        eos_positions = (
            generated
            == tokenizer.eos_token_id
        ).nonzero(
            as_tuple=False
        )

        if len(
            eos_positions
        ) > 0:

            end = (
                int(
                    eos_positions[0].item()
                )
                + 1
            )

            generated = generated[
                :end
            ]

        prompt = torch.tensor(
            encoded[i],
            dtype=torch.long,
        )

        results.append(
            (
                prompt.cpu(),
                generated.detach().cpu(),
            )
        )

    return results


# ============================================================
# LOG PROBS
# ============================================================

def generated_logprobs(
    model,
    prompt_ids,
    generated_ids,
    requires_grad,
):

    """
    Computes log probabilities only for generated assistant tokens.
    """

    prompt_ids = prompt_ids.to(
        DEVICE
    )

    generated_ids = (
        generated_ids.to(
            DEVICE
        )
    )

    if (
        generated_ids.numel()
        == 0
    ):

        return torch.empty(
            0,
            device=DEVICE,
        )

    full_ids = torch.cat(
        [
            prompt_ids,
            generated_ids,
        ],
        dim=0,
    )

    model_input = (
        full_ids[:-1]
        .unsqueeze(0)
    )

    targets = full_ids[
        1:
    ]

    if requires_grad:

        outputs = model(
            input_ids=model_input,
            use_cache=False,
        )

    else:

        with torch.no_grad():

            outputs = model(
                input_ids=model_input,
                use_cache=False,
            )

    logits = (
        outputs.logits[0]
        .float()
    )

    log_probs = (
        F.log_softmax(
            logits,
            dim=-1,
        )
    )

    selected = (
        log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    # token before generated token #1
    start = (
        len(prompt_ids)
        - 1
    )

    end = (
        start
        + len(generated_ids)
    )

    return selected[
        start:end
    ]


# ============================================================
# MEMORY TRUNCATION
# ============================================================

def truncate_memory(
    tokenizer,
    memory,
):

    ids = tokenizer.encode(
        memory,
        add_special_tokens=False,
    )

    if (
        len(ids)
        <= MAX_MEMORY_INPUT_TOKENS
    ):

        return memory

    ids = ids[
        -MAX_MEMORY_INPUT_TOKENS:
    ]

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )


# ============================================================
# DOCUMENT CHUNKS
# ============================================================

def make_chunks(example):

    docs = get_documents(
        example
    )

    chunks = []

    for i in range(
        0,
        len(docs),
        DOCS_PER_SEGMENT,
    ):

        chunks.append(
            docs[
                i:
                i + DOCS_PER_SEGMENT
            ]
        )

    return chunks[
        :MAX_COMPACTIONS
    ]


# ============================================================
# BATCHED TRAJECTORY ROLLOUT
# ============================================================

def rollout_batch(
    model,
    tokenizer,
    examples,
    training,
):

    """
    Example:

        20 questions
             ↓
        Summary round 1 batch=20
             ↓
        Summary round 2 batch=20
             ↓
        Summary round 3 batch=20
             ↓
        Answer batch=20

    If NUM_ROLLOUTS=2 during training:
        internally expands 20 → 40 trajectories.

    Generation is batched.
    """

    expanded = []

    for example in examples:

        repeat = (
            NUM_ROLLOUTS
            if training
            else 1
        )

        for _ in range(
            repeat
        ):

            expanded.append(
                {
                    "example":
                        example,

                    "question":
                        example[
                            "question"
                        ],

                    "gold":
                        example[
                            "answer"
                        ],

                    "chunks":
                        make_chunks(
                            example
                        ),

                    "memory":
                        "",

                    "segments":
                        [],
                }
            )

    if len(expanded) == 0:
        return []

    max_rounds = max(
        len(x["chunks"])
        for x in expanded
    )

    # ========================================================
    # COMPACTION ROUNDS
    # ========================================================

    for round_idx in range(
        max_rounds
    ):

        active_indices = []

        message_batch = []

        for idx, state in enumerate(
            expanded
        ):

            if (
                round_idx
                >= len(
                    state["chunks"]
                )
            ):
                continue

            active_indices.append(
                idx
            )

            message_batch.append(
                build_summary_messages(
                    question=state[
                        "question"
                    ],

                    old_memory=state[
                        "memory"
                    ],

                    new_docs=state[
                        "chunks"
                    ][round_idx],
                )
            )

        if len(
            active_indices
        ) == 0:

            continue

        # ----------------------------------------------------
        # For training with NUM_ROLLOUTS=2:
        #
        # 20 prompts => 40 trajectories.
        #
        # Avoid generation OOM by splitting generation calls
        # into chunks of TRAIN_BATCH_SIZE=20.
        # ----------------------------------------------------

        generated_all = []

        generation_bs = (
            TRAIN_BATCH_SIZE
            if training
            else EVAL_BATCH_SIZE
        )

        for start in range(
            0,
            len(message_batch),
            generation_bs,
        ):

            sub_messages = (
                message_batch[
                    start:
                    start
                    + generation_bs
                ]
            )

            sub_generated = (
                generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    messages_batch=sub_messages,
                    max_new_tokens=
                        MAX_SUMMARY_TOKENS,
                    do_sample=training,
                )
            )

            generated_all.extend(
                sub_generated
            )

        for local_idx, (
            prompt_ids,
            generated_ids,
        ) in enumerate(
            generated_all
        ):

            state_idx = (
                active_indices[
                    local_idx
                ]
            )

            state = expanded[
                state_idx
            ]

            memory = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            memory = truncate_memory(
                tokenizer,
                memory,
            )

            state[
                "memory"
            ] = memory

            if training:

                # --------------------------------------------
                # Store behavior-policy log probabilities.
                # Detached reference for PPO ratio.
                # --------------------------------------------

                old_lp = (
                    generated_logprobs(
                        model,
                        prompt_ids,
                        generated_ids,
                        requires_grad=False,
                    )
                    .detach()
                    .cpu()
                )

                state[
                    "segments"
                ].append(
                    Segment(
                        prompt_ids=
                            prompt_ids,

                        generated_ids=
                            generated_ids,

                        old_logprobs=
                            old_lp,

                        kind=
                            f"summary_{round_idx+1}",
                    )
                )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Nothing from the previous document chunk remains
        # except state["memory"].
        #
        # That is the actual context-compaction behavior.
        # ----------------------------------------------------

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    answer_messages = []

    for state in expanded:

        answer_messages.append(
            build_answer_messages(
                question=state[
                    "question"
                ],

                memory=state[
                    "memory"
                ],
            )
        )

    answer_generated_all = []

    generation_bs = (
        TRAIN_BATCH_SIZE
        if training
        else EVAL_BATCH_SIZE
    )

    for start in range(
        0,
        len(answer_messages),
        generation_bs,
    ):

        sub_messages = (
            answer_messages[
                start:
                start
                + generation_bs
            ]
        )

        outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            messages_batch=sub_messages,
            max_new_tokens=
                MAX_ANSWER_TOKENS,
            do_sample=training,
        )

        answer_generated_all.extend(
            outputs
        )

    trajectories = []

    for state, (
        prompt_ids,
        generated_ids,
    ) in zip(
        expanded,
        answer_generated_all,
    ):

        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        # Short answer only.
        if "\n" in answer:

            answer = answer.split(
                "\n"
            )[0].strip()

        reward = answer_f1(
            answer,
            state["gold"],
        )

        if training:

            old_lp = (
                generated_logprobs(
                    model,
                    prompt_ids,
                    generated_ids,
                    requires_grad=False,
                )
                .detach()
                .cpu()
            )

            state[
                "segments"
            ].append(
                Segment(
                    prompt_ids=
                        prompt_ids,

                    generated_ids=
                        generated_ids,

                    old_logprobs=
                        old_lp,

                    kind="answer",
                )
            )

        trajectories.append(
            Trajectory(
                segments=
                    state["segments"],

                reward=
                    reward,

                answer=
                    answer,

                memory=
                    state["memory"],

                gold=
                    state["gold"],

                question=
                    state["question"],
            )
        )

    return trajectories


# ============================================================
# GROUP ADVANTAGES
# ============================================================

def compute_group_advantages(
    trajectories,
    num_prompts,
):

    """
    trajectories layout:

        prompt 1 rollout 1
        prompt 1 rollout 2

        prompt 2 rollout 1
        prompt 2 rollout 2

        ...

    Returns one advantage per trajectory.
    """

    advantages = []

    ptr = 0

    for _ in range(
        num_prompts
    ):

        group = trajectories[
            ptr:
            ptr + NUM_ROLLOUTS
        ]

        rewards = torch.tensor(
            [
                x.reward
                for x in group
            ],
            dtype=torch.float32,
        )

        mean = rewards.mean()

        std = rewards.std(
            unbiased=False
        )

        if std.item() > 1e-6:

            group_adv = (
                rewards
                - mean
            ) / (
                std
                + 1e-6
            )

        else:

            # Both rollouts have same reward:
            # no relative learning signal.
            group_adv = torch.zeros_like(
                rewards
            )

        advantages.extend(
            group_adv.tolist()
        )

        ptr += NUM_ROLLOUTS

    return advantages


# ============================================================
# PPO TOKEN LOSS
# ============================================================

def trajectory_loss(
    model,
    trajectory,
    advantage,
):

    """
    Final task advantage is applied to every generated assistant token:

        summary1
        summary2
        summary3
        final answer

    Thus summary generation + answer generation are jointly trained.
    """

    if (
        abs(advantage)
        < 1e-12
    ):

        return None

    total_loss = None
    total_tokens = 0

    advantage_tensor = torch.tensor(
        advantage,
        dtype=torch.float32,
        device=DEVICE,
    )

    for segment in trajectory.segments:

        if (
            segment.generated_ids.numel()
            == 0
        ):
            continue

        new_lp = generated_logprobs(
            model,
            segment.prompt_ids,
            segment.generated_ids,
            requires_grad=True,
        )

        old_lp = (
            segment.old_logprobs
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # PPO ratio
        # ----------------------------------------------------

        log_ratio = (
            new_lp
            - old_lp
        )

        # Numerical safety
        log_ratio = torch.clamp(
            log_ratio,
            -10,
            10,
        )

        ratio = torch.exp(
            log_ratio
        )

        unclipped = (
            ratio
            * advantage_tensor
        )

        clipped_ratio = (
            torch.clamp(
                ratio,
                1.0 - PPO_CLIP,
                1.0 + PPO_CLIP,
            )
        )

        clipped = (
            clipped_ratio
            * advantage_tensor
        )

        token_objective = (
            torch.minimum(
                unclipped,
                clipped,
            )
        )

        segment_loss = (
            -token_objective.sum()
        )

        if total_loss is None:

            total_loss = segment_loss

        else:

            total_loss = (
                total_loss
                + segment_loss
            )

        total_tokens += (
            new_lp.numel()
        )

    if (
        total_loss is None
        or total_tokens == 0
    ):

        return None

    # Token-normalized loss
    return (
        total_loss
        / total_tokens
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    print(
        "Loading model..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,

            torch_dtype=
                DTYPE,

            device_map={
                "": 0
            },

            trust_remote_code=True,
        )
    )

    adapter_config = os.path.join(
        PREVIOUS_ADAPTER,
        "adapter_config.json",
    )

    if os.path.exists(
        adapter_config
    ):

        print(
            "\nUsing previous adapter:"
        )

        print(
            PREVIOUS_ADAPTER
        )

        model = (
            PeftModel
            .from_pretrained(
                model,
                PREVIOUS_ADAPTER,
                is_trainable=True,
            )
        )

    else:

        print(
            "\nPrevious adapter not found."
        )

        print(
            "Creating fresh LoRA."
        )

        peft_config = LoraConfig(
            r=4,

            lora_alpha=8,

            lora_dropout=0.0,

            target_modules=[
                "q_proj",
                "v_proj",
            ],

            bias="none",

            task_type=
                "CAUSAL_LM",
        )

        model = get_peft_model(
            model,
            peft_config,
        )

    # generation uses cache
    model.config.use_cache = True

    model.print_trainable_parameters()

    return (
        model,
        tokenizer,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    dataset,
    name,
):

    model.eval()

    all_ems = []
    all_f1s = []

    output_examples = []

    print(
        "\n"
        + "=" * 72
    )

    print(
        name
    )

    print(
        "=" * 72
    )

    for batch_start in range(
        0,
        len(dataset),
        EVAL_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start
            + EVAL_BATCH_SIZE,
            len(dataset),
        )

        examples = [
            dataset[i]
            for i in range(
                batch_start,
                batch_end,
            )
        ]

        trajectories = (
            rollout_batch(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                training=False,
            )
        )

        for trajectory in trajectories:

            em = exact_match(
                trajectory.answer,
                trajectory.gold,
            )

            f1 = answer_f1(
                trajectory.answer,
                trajectory.gold,
            )

            all_ems.append(
                em
            )

            all_f1s.append(
                f1
            )

            output_examples.append(
                {
                    "question":
                        trajectory.question,

                    "gold":
                        trajectory.gold,

                    "prediction":
                        trajectory.answer,

                    "memory":
                        trajectory.memory,

                    "em":
                        em,

                    "f1":
                        f1,
                }
            )

        completed = len(
            all_ems
        )

        mean_em = (
            sum(all_ems)
            / completed
        )

        mean_f1 = (
            sum(all_f1s)
            / completed
        )

        print(
            f"[{completed}/{len(dataset)}] "
            f"EM={mean_em:.4f} "
            f"F1={mean_f1:.4f}"
        )

    metrics = {
        "N":
            len(all_ems),

        "EM":
            sum(all_ems)
            / len(all_ems),

        "F1":
            sum(all_f1s)
            / len(all_f1s),
    }

    print(
        "\nFinal:",
        metrics,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"{name}.json",
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
                    output_examples,
            },

            f,

            ensure_ascii=False,
            indent=2,
        )

    return metrics


# ============================================================
# TRAIN
# ============================================================

def train(
    model,
    tokenizer,
    dataset,
):

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = (
        torch.optim.AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            betas=(0.9, 0.99),
            weight_decay=0.01,
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    global_step = 0

    for epoch in range(
        EPOCHS
    ):

        indices = list(
            range(
                len(dataset)
            )
        )

        rng = random.Random(
            SEED + epoch
        )

        rng.shuffle(
            indices
        )

        for batch_start in range(
            0,
            len(indices),
            TRAIN_BATCH_SIZE,
        ):

            batch_indices = (
                indices[
                    batch_start:
                    batch_start
                    + TRAIN_BATCH_SIZE
                ]
            )

            examples = [
                dataset[i]
                for i in batch_indices
            ]

            # =================================================
            # 1. Batched rollout
            # =================================================

            model.eval()

            trajectories = (
                rollout_batch(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    training=True,
                )
            )

            # =================================================
            # 2. Group-relative advantage
            # =================================================

            advantages = (
                compute_group_advantages(
                    trajectories=
                        trajectories,

                    num_prompts=
                        len(examples),
                )
            )

            rewards = [
                x.reward
                for x
                in trajectories
            ]

            reward_mean = (
                sum(rewards)
                / len(rewards)
            )

            nonzero_adv = sum(
                1
                for x in advantages
                if abs(x) > 1e-8
            )

            # =================================================
            # 3. One optimizer update for the prompt batch
            #
            # Physical forward/backward is trajectory-wise
            # to prevent 4090 OOM.
            #
            # Mathematically normalize over batch.
            # =================================================

            model.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            valid_losses = 0

            accumulated_loss_value = 0.0

            # First know how many trajectories carry signal.
            active_count = sum(
                1
                for x
                in advantages
                if abs(x) > 1e-8
            )

            if active_count > 0:

                for trajectory, advantage in zip(
                    trajectories,
                    advantages,
                ):

                    if (
                        abs(advantage)
                        <= 1e-8
                    ):
                        continue

                    loss = trajectory_loss(
                        model=model,
                        trajectory=trajectory,
                        advantage=advantage,
                    )

                    if loss is None:
                        continue

                    # Normalize across active trajectories
                    scaled_loss = (
                        loss
                        / active_count
                    )

                    scaled_loss.backward()

                    accumulated_loss_value += float(
                        scaled_loss.detach()
                    )

                    valid_losses += 1

                if valid_losses > 0:

                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        MAX_GRAD_NORM,
                    )

                    optimizer.step()

                    global_step += 1

            batch_number = (
                batch_start
                // TRAIN_BATCH_SIZE
                + 1
            )

            total_batches = (
                len(indices)
                + TRAIN_BATCH_SIZE
                - 1
            ) // TRAIN_BATCH_SIZE

            print(
                f"[epoch {epoch+1}] "
                f"batch "
                f"{batch_number}/{total_batches} | "
                f"prompts={len(examples)} | "
                f"trajectories={len(trajectories)} | "
                f"reward={reward_mean:.4f} | "
                f"active_adv={nonzero_adv}/"
                f"{len(advantages)} | "
                f"loss={accumulated_loss_value:.6f}"
            )

            del trajectories

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Device:",
        DEVICE,
    )

    print(
        "dtype:",
        DTYPE,
    )

    print(
        "Train generation batch:",
        TRAIN_BATCH_SIZE,
    )

    print(
        "Eval generation batch:",
        EVAL_BATCH_SIZE,
    )

    print(
        "Rollouts per prompt:",
        NUM_ROLLOUTS,
    )

    train_dataset, eval_dataset = (
        load_same_data()
    )

    model, tokenizer = (
        load_model()
    )

    # ========================================================
    # BEFORE
    # ========================================================

    if EVAL_BEFORE:

        before = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            name=
                "before_segmented_training",
        )

    else:

        before = None

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SEGMENTED COMPACTION RL TRAINING"
    )

    print(
        "=" * 72
    )

    model = train(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print(
        "\nSaved:",
        OUTPUT_DIR,
    )

    # ========================================================
    # AFTER
    # ========================================================

    if EVAL_AFTER:

        after = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            name=
                "after_segmented_training",
        )

    else:

        after = None

    # ========================================================
    # SUMMARY
    # ========================================================

    comparison = {
        "before":
            before,

        "after":
            after,

        "train_n":
            TRAIN_N,

        "eval_n":
            EVAL_N,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "eval_batch_size":
            EVAL_BATCH_SIZE,

        "num_rollouts":
            NUM_ROLLOUTS,

        "seed":
            SEED,
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "comparison.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        "Before:",
        before,
    )

    print(
        "After :",
        after,
    )


if __name__ == "__main__":

    main()
