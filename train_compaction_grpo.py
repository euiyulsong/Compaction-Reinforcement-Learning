# train_compaction_grpo.py

import os
import re
import json
import random
import string
import inspect
import importlib.util
import collections

import torch

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

TRAIN_N = 1000
EVAL_N = 200

SEED = 42

OUTPUT_DIR = "./qwen3_compaction_grpo"

# ------------------------------------------------------------
# RTX 4090 24GB-safe-ish starting point
# ------------------------------------------------------------

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

# GRPO needs >= 2 generations
NUM_GENERATIONS = 2

# We manually truncate documents before giving them to TRL.
MAX_DOCUMENT_TOKENS = 1200

MAX_COMPLETION_LENGTH = 256

LEARNING_RATE = 2e-6

TEMPERATURE = 0.8
TOP_P = 0.95

FORMAT_REWARD_WEIGHT = 0.05

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Environment
# ============================================================

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("PyTorch:", torch.__version__)

try:
    import trl
    print("TRL:", trl.__version__)
except Exception:
    pass

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported()
    )

HAS_VLLM = (
    importlib.util.find_spec("vllm")
    is not None
)

print("vLLM installed:", HAS_VLLM)


# ============================================================
# Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# generation tends to behave more predictably this way
tokenizer.padding_side = "left"


# ============================================================
# HotpotQA metric
# ============================================================

def normalize_answer(s):

    if s is None:
        return ""

    s = str(s)

    def lower(text):
        return text.lower()

    def remove_punc(text):

        exclude = set(
            string.punctuation
        )

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

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

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(s)
            )
        )
    )


def exact_match(prediction, ground_truth):

    return float(
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


def answer_f1(prediction, ground_truth):

    pred = normalize_answer(
        prediction
    )

    gold = normalize_answer(
        ground_truth
    )

    # HotpotQA yes/no cases
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

    if len(pred_tokens) == 0:
        return float(
            len(gold_tokens) == 0
        )

    if len(gold_tokens) == 0:
        return float(
            len(pred_tokens) == 0
        )

    common = (
        collections.Counter(
            pred_tokens
        )
        &
        collections.Counter(
            gold_tokens
        )
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
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You solve multi-hop question answering tasks.

While reading the evidence, create a compact internal memory that
preserves only information useful for answering the question.

Your output MUST follow exactly this format:

<MEMORY>
Compact useful facts and intermediate relationships.
</MEMORY>

<REASONING>
Reason from the compact memory.
</REASONING>

<ANSWER>
short final answer
</ANSWER>

The MEMORY section receives no independent correctness reward.
Its purpose is only to help produce the correct final answer.
""".strip()


def truncate_documents(
    documents,
    max_tokens,
):

    """
    Keep complete documents until the token budget is nearly full.

    This is preferable to blindly truncating the complete prompt,
    because the question / instructions are never removed.
    """

    kept = []
    used = 0

    for document in documents:

        ids = tokenizer.encode(
            document,
            add_special_tokens=False,
        )

        if (
            used + len(ids)
            <= max_tokens
        ):

            kept.append(
                document
            )

            used += len(ids)

            continue

        remaining = (
            max_tokens - used
        )

        if remaining > 50:

            partial_ids = ids[
                :remaining
            ]

            partial = tokenizer.decode(
                partial_ids,
                skip_special_tokens=True,
            )

            kept.append(
                partial
            )

        break

    return "\n\n".join(
        kept
    )


def make_chat_prompt(
    question,
    documents,
):

    user_content = f"""
Question:
{question}

Evidence documents:
{documents}

Use the evidence to solve the question.
First compact the useful state into <MEMORY>, then reason,
then put ONLY the short final answer inside <ANSWER>.
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    # Qwen3 tokenizer supports enable_thinking,
    # but keep compatibility with tokenizer versions
    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


# ============================================================
# Dataset
# ============================================================

def convert_hotpot_example(
    example,
):

    titles = (
        example["context"]["title"]
    )

    sentence_lists = (
        example["context"]["sentences"]
    )

    docs = []

    for title, sentences in zip(
        titles,
        sentence_lists,
    ):

        docs.append(
            f"[Document: {title}]\n"
            + " ".join(sentences)
        )

    docs = truncate_documents(
        docs,
        MAX_DOCUMENT_TOKENS,
    )

    prompt = make_chat_prompt(
        example["question"],
        docs,
    )

    return {
        "prompt": prompt,
        "answer": example["answer"],
        "question": example["question"],
    }


def load_data():

    print("\n" + "=" * 70)
    print("LOADING HOTPOTQA")
    print("=" * 70)

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    train_raw = (
        ds["train"]
        .shuffle(seed=SEED)
        .select(
            range(TRAIN_N)
        )
    )

    eval_raw = (
        ds["validation"]
        .shuffle(seed=SEED)
        .select(
            range(EVAL_N)
        )
    )

    print(
        "Building train prompts..."
    )

    train_rows = [
        convert_hotpot_example(x)
        for x in train_raw
    ]

    print(
        "Building eval prompts..."
    )

    eval_rows = [
        convert_hotpot_example(x)
        for x in eval_raw
    ]

    train_dataset = (
        Dataset.from_list(
            train_rows
        )
    )

    eval_dataset = (
        Dataset.from_list(
            eval_rows
        )
    )

    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Eval :",
        len(eval_dataset)
    )

    sample_tokens = len(
        tokenizer.encode(
            train_dataset[0][
                "prompt"
            ],
            add_special_tokens=False,
        )
    )

    print(
        "Example prompt tokens:",
        sample_tokens
    )

    return (
        train_dataset,
        eval_dataset,
    )


# ============================================================
# Completion parsing
# ============================================================

def completion_to_text(
    completion,
):

    # Most TRL versions return str for
    # standard/non-conversational prompts.
    if isinstance(
        completion,
        str,
    ):
        return completion

    # Compatibility with conversational completion.
    if isinstance(
        completion,
        list,
    ):

        texts = []

        for item in completion:

            if isinstance(
                item,
                dict,
            ):

                texts.append(
                    str(
                        item.get(
                            "content",
                            "",
                        )
                    )
                )

            else:

                texts.append(
                    str(item)
                )

        return "\n".join(
            texts
        )

    if isinstance(
        completion,
        dict,
    ):

        return str(
            completion.get(
                "content",
                completion,
            )
        )

    return str(completion)


def extract_final_answer(
    text,
):

    if text is None:
        return ""

    text = str(text)

    # Preferred format
    match = re.search(
        r"<ANSWER>\s*(.*?)\s*</ANSWER>",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if match:

        result = (
            match.group(1)
            .strip()
        )

        return (
            result
            .split("\n")[0]
            .strip()
        )

    # fallback
    match = re.search(
        r"ANSWER\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return (
            match.group(1)
            .split("\n")[0]
            .strip()
        )

    # last non-empty line fallback
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    if lines:
        return lines[-1]

    return ""


# ============================================================
# Reward
# ============================================================

def compaction_reward(
    completions,
    answer,
    **kwargs,
):

    """
    One terminal reward.

    There is NO gold summary and NO independent summary-quality
    supervision.

    Final answer F1 is the primary reward.

    Because GRPO optimizes the entire generated completion,
    MEMORY + REASONING + ANSWER tokens all receive credit
    from this terminal reward.

    This is the 1-GPU approximation of the CompactionRL idea.
    """

    rewards = []

    for completion, gold in zip(
        completions,
        answer,
    ):

        text = completion_to_text(
            completion
        )

        prediction = (
            extract_final_answer(
                text
            )
        )

        f1 = answer_f1(
            prediction,
            gold,
        )

        # tiny formatting reward
        has_memory = bool(
            re.search(
                r"<MEMORY>.*?</MEMORY>",
                text,
                flags=(
                    re.IGNORECASE
                    | re.DOTALL
                ),
            )
        )

        has_answer = bool(
            re.search(
                r"<ANSWER>.*?</ANSWER>",
                text,
                flags=(
                    re.IGNORECASE
                    | re.DOTALL
                ),
            )
        )

        format_bonus = (
            FORMAT_REWARD_WEIGHT
            if (
                has_memory
                and has_answer
            )
            else 0.0
        )

        reward = (
            float(f1)
            + format_bonus
        )

        rewards.append(
            reward
        )

    return rewards


# ============================================================
# Compatibility helper
# ============================================================

def build_grpo_config():

    """
    TRL has changed GRPOConfig fields across versions.

    Only arguments that actually exist in the installed version
    are passed.
    """

    signature = inspect.signature(
        GRPOConfig.__init__
    )

    supported = set(
        signature.parameters.keys()
    )

    desired = {

        # standard Trainer args
        "output_dir":
            OUTPUT_DIR,

        "learning_rate":
            LEARNING_RATE,

        "adam_beta1":
            0.9,

        "adam_beta2":
            0.99,

        "weight_decay":
            0.01,

        "max_grad_norm":
            1.0,

        "num_train_epochs":
            1,

        "per_device_train_batch_size":
            PER_DEVICE_BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION,

        "logging_steps":
            5,

        "save_steps":
            50,

        "save_total_limit":
            2,

        "report_to":
            "none",

        "remove_unused_columns":
            False,

        "seed":
            SEED,

        # GRPO
        "num_generations":
            NUM_GENERATIONS,

        "num_iterations":
            1,

        "epsilon":
            0.2,

        "beta":
            0.01,

        # generation
        "max_completion_length":
            MAX_COMPLETION_LENGTH,

        "temperature":
            TEMPERATURE,

        "top_p":
            TOP_P,

        # precision
        "bf16":
            (
                torch.cuda.is_available()
                and
                torch.cuda.is_bf16_supported()
            ),

        "fp16":
            (
                torch.cuda.is_available()
                and
                not torch.cuda.is_bf16_supported()
            ),

        # memory
        "gradient_checkpointing":
            True,

        "gradient_checkpointing_kwargs":
            {
                "use_reentrant": False
            },

        # vLLM colocate
        "use_vllm":
            HAS_VLLM,

        "vllm_mode":
            "colocate",

        # Conservative for RTX 4090.
        "vllm_gpu_memory_utilization":
            0.25,

        "vllm_enable_sleep_mode":
            True,
    }

    final_kwargs = {}
    skipped = {}

    for key, value in (
        desired.items()
    ):

        if key in supported:

            final_kwargs[key] = value

        else:

            skipped[key] = value

    print("\n" + "=" * 70)
    print("GRPO CONFIG COMPATIBILITY")
    print("=" * 70)

    print(
        "Supported desired args:"
    )

    for key in final_kwargs:
        print("  +", key)

    if skipped:

        print(
            "\nUnsupported args "
            "(automatically skipped):"
        )

        for key in skipped:
            print("  -", key)

    return GRPOConfig(
        **final_kwargs
    )


# ============================================================
# GRPOTrainer compatibility
# ============================================================

def build_trainer(
    training_args,
    train_dataset,
):

    signature = inspect.signature(
        GRPOTrainer.__init__
    )

    supported = set(
        signature.parameters.keys()
    )

    kwargs = {
        "model":
            MODEL_NAME,

        "args":
            training_args,

        "train_dataset":
            train_dataset,

        "reward_funcs":
            compaction_reward,
    }

    # tokenizer argument changed name in HF Trainer ecosystem
    if (
        "processing_class"
        in supported
    ):

        kwargs[
            "processing_class"
        ] = tokenizer

    elif (
        "tokenizer"
        in supported
    ):

        kwargs[
            "tokenizer"
        ] = tokenizer

    # LoRA
    if (
        "peft_config"
        in supported
    ):

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,

            lora_dropout=0.0,

            bias="none",

            task_type="CAUSAL_LM",

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )

        kwargs[
            "peft_config"
        ] = peft_config

    else:

        print(
            "WARNING: installed GRPOTrainer "
            "does not expose peft_config."
        )

    print("\nCreating GRPOTrainer...")

    return GRPOTrainer(
        **kwargs
    )


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    trainer,
    eval_dataset,
):

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model = trainer.model

    model.eval()

    device = next(
        model.parameters()
    ).device

    em_scores = []
    f1_scores = []

    outputs_json = []

    for i in range(
        len(eval_dataset)
    ):

        row = eval_dataset[i]

        prompt = row["prompt"]

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        prompt_length = (
            encoded[
                "input_ids"
            ].shape[1]
        )

        generated = model.generate(
            **encoded,

            max_new_tokens=(
                MAX_COMPLETION_LENGTH
            ),

            do_sample=False,

            use_cache=True,

            pad_token_id=(
                tokenizer.pad_token_id
            ),

            eos_token_id=(
                tokenizer.eos_token_id
            ),
        )

        new_tokens = generated[
            0,
            prompt_length:
        ]

        text = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        prediction = (
            extract_final_answer(
                text
            )
        )

        gold = row["answer"]

        em = exact_match(
            prediction,
            gold,
        )

        f1 = answer_f1(
            prediction,
            gold,
        )

        em_scores.append(em)
        f1_scores.append(f1)

        outputs_json.append({
            "question":
                row["question"],

            "gold":
                gold,

            "prediction":
                prediction,

            "em":
                em,

            "f1":
                f1,

            "generation":
                text,
        })

        if (
            (i + 1) % 20 == 0
            or
            i == 0
        ):

            print(
                f"[{i+1:03d}/"
                f"{len(eval_dataset)}] "
                f"EM="
                f"{sum(em_scores)/len(em_scores):.4f} "
                f"F1="
                f"{sum(f1_scores)/len(f1_scores):.4f}"
            )

            print(
                "  gold:",
                gold
            )

            print(
                "  pred:",
                prediction
            )

    metrics = {
        "EM": (
            sum(em_scores)
            / len(em_scores)
        ),

        "F1": (
            sum(f1_scores)
            / len(f1_scores)
        ),

        "num_examples":
            len(eval_dataset),
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        OUTPUT_DIR,
        "eval_results.json",
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
                    outputs_json,
            },

            f,

            indent=2,

            ensure_ascii=False,
        )

    print("\nFINAL EVALUATION")

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        "\nSaved:",
        path
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    (
        train_dataset,
        eval_dataset,
    ) = load_data()

    training_args = (
        build_grpo_config()
    )

    trainer = build_trainer(
        training_args,
        train_dataset,
    )

    print("\n" + "=" * 70)
    print("TRAIN")
    print("=" * 70)

    print(
        "Training examples:",
        len(train_dataset)
    )

    print(
        "Generations / prompt:",
        NUM_GENERATIONS
    )

    print(
        "Effective prompt batch:",
        (
            PER_DEVICE_BATCH_SIZE
            * GRADIENT_ACCUMULATION
        )
    )

    print(
        "Effective generated sequences:",
        (
            PER_DEVICE_BATCH_SIZE
            * GRADIENT_ACCUMULATION
            * NUM_GENERATIONS
        )
    )

    trainer.train()

    print("\nSaving model...")

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    # free some fragmented cache before eval
    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    evaluate(
        trainer,
        eval_dataset,
    )


if __name__ == "__main__":
    main()
