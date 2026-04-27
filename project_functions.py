#!/usr/bin/env python
# coding: utf-8

import os
import json
import random
import string
import re
import contextlib
import io
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR     = Path("/home/pranav/Documents/comp6881project/data")
PROC_DIR     = Path("/home/pranav/Documents/comp6881project/data/processed")
FEATURES_DIR = Path("/home/pranav/Documents/comp6881project/data/features")
RAW_DIR      = DATA_DIR / "raw"
TRAIN_FILE   = PROC_DIR / "train_labeled.json"
VAL_FILE     = PROC_DIR / "val_labeled.json"
TEST_FILE    = PROC_DIR / "test_labeled.json"

N_TRIVIAQA    = 2500
N_SQUAD_ANS   = 1250
N_SQUAD_UNANS = 1250
F1_THRESHOLD  = 0.4

MAX_NEW_TOKENS   = 50
K_SAMPLES        = 5
TEMPERATURE      = 0.7
CHECKPOINT_EVERY = 50
PCA_COMPONENTS   = 1024

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16

NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}

ABSTENTION_PHRASES = [
    "i cannot answer this",
    "i cannot answer",
    "i don't know",
    "i do not know",
    "the passage does not",
    "the passage doesn't",
    "there is no information",
    "there is no mention",
    "i cannot find",
    "i do not have information",
]

def load_triviaqa(n_samples: int, seed: int = SEED) -> List[Dict]:
    """Load TriviaQA (rc.wikipedia.nocontext) and format into a standardised dict.

    The model is queried in closed-book mode with no context.

    Args:
        n_samples: Number of examples to load from the TriviaQA validation split.
        seed: Random seed for shuffling before selection.

    Returns:
        List of dicts with keys: id, source, question, references, context.
    """
    print("Loading TriviaQA (rc.wikipedia.nocontext) …")
    raw = load_dataset(
        "trivia_qa",
        "unfiltered.nocontext",
        split="validation",
        cache_dir=str(RAW_DIR),
        trust_remote_code=True,
    )
    raw = raw.shuffle(seed=seed).select(range(n_samples))

    examples = []
    for item in tqdm(raw, desc="TriviaQA"):
        values  = item["answer"]["normalized_value"]
        aliases = item["answer"]["normalized_aliases"]

        def is_clean_alias(s):
            if not s or len(s) < 2:
                return False
            if re.match(r'^[0-9a-f]{6}$', s) or s.startswith('rgb('):
                return False
            if re.match(r'^iso \d', s):
                return False
            if re.match(r'^[0-9\s\-]+$', s) and values and not re.match(r'^[0-9\s\-]+$', values):
                return False
            return True

        clean_aliases = [a for a in aliases if is_clean_alias(a)]
        references    = list(dict.fromkeys([values] + clean_aliases))
        if not references:
            continue

        examples.append({
            "id":         item["question_id"],
            "source":     "triviaqa",
            "question":   item["question"],
            "references": references,
            "context":    None,
        })
    return examples


def load_squad2(n_answerable: int, n_unanswerable: int, seed: int = SEED) -> List[Dict]:
    """Load SQuAD 2.0 validation split.

    Answerable examples: references extracted from the 'answers' field.
    Unanswerable examples: references = [] (any non-empty model output = hallucination).

    Args:
        n_answerable: Number of answerable examples to sample.
        n_unanswerable: Number of unanswerable examples to sample.
        seed: Random seed for sampling.

    Returns:
        List of dicts with keys: id, source, question, references, context, is_unanswerable.
    """
    raw = load_dataset("rajpurkar/squad_v2", split="validation", cache_dir=str(RAW_DIR))

    answerable   = [ex for ex in raw if len(ex["answers"]["text"]) > 0]
    unanswerable = [ex for ex in raw if len(ex["answers"]["text"]) == 0]

    rng          = random.Random(seed)
    answerable   = rng.sample(answerable,   min(n_answerable,   len(answerable)))
    unanswerable = rng.sample(unanswerable, min(n_unanswerable, len(unanswerable)))

    print(f"  Answerable:   {len(answerable)}")
    print(f"  Unanswerable: {len(unanswerable)}")

    examples = []
    for item in tqdm(answerable + unanswerable, desc="SQuAD 2.0"):
        is_unans = len(item["answers"]["text"]) == 0
        examples.append({
            "id":             item["id"],
            "source":         "squad2",
            "question":       item["question"],
            "references":     item["answers"]["text"],
            "context":        item["context"],
            "is_unanswerable": is_unans,
        })
    return examples


def assign_placeholder_label(ex: Dict) -> Dict:
    """Mark each example with its hallucination condition type.

    Actual binary label (0/1) is assigned after model generation by comparing
    model_answer against references.

    Args:
        ex: Example dict with keys 'source' and optionally 'is_unanswerable'.

    Returns:
        The same dict with 'condition', 'model_answer', 'label', and 'token_f1' set.
    """
    if ex["source"] == "triviaqa":
        ex["condition"] = "closed_book"
    elif ex.get("is_unanswerable"):
        ex["condition"] = "unanswerable"
    else:
        ex["condition"] = "answerable"

    ex["model_answer"] = None
    ex["label"]        = None
    ex["token_f1"]     = None
    return ex


def make_splits(all_examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Stratified 70/10/20 train/val/test split preserving condition balance.

    Args:
        all_examples: Full list of labelled example dicts.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    strata       = [f"{ex['source']}_{ex['condition']}" for ex in all_examples]
    train_val, test = train_test_split(
        all_examples, test_size=0.20, stratify=strata, random_state=SEED
    )
    strata_tv    = [f"{ex['source']}_{ex['condition']}" for ex in train_val]
    train, val   = train_test_split(
        train_val, test_size=0.125, stratify=strata_tv, random_state=SEED
    )
    return {"train": train, "val": val, "test": test}

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles and punctuation, and normalise number words to digits.

    Args:
        s: Answer string to normalise.

    Returns:
        Normalised answer string.
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    tokens = [NUM_WORDS.get(t, t) for t in s.split()]
    return ' '.join(tokens)


def token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute the best token-level F1 score between a prediction and reference answers.

    Tokenises both strings after normalisation and uses Counter intersection to
    handle duplicate tokens. Returns the maximum F1 across all references.

    Args:
        prediction: Model-generated answer string.
        ground_truths: One or more reference answer strings.

    Returns:
        Best F1 score in [0, 1] across all references. Returns 0.0 if no overlap.
    """
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common    = Counter(pred_tokens) & Counter(gt_tokens)
        n_common  = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gt_tokens)
        f1        = (2 * precision * recall) / (precision + recall)
        best_f1   = max(best_f1, f1)
    return best_f1


def is_abstention(text: str) -> bool:
    """Return True if the model correctly refused to answer.

    Args:
        text: Model-generated answer string.

    Returns:
        True if any known abstention phrase is present, False otherwise.
    """
    return any(phrase in text.lower().strip() for phrase in ABSTENTION_PHRASES)


def label_example(ex: Dict, f1_threshold: float = F1_THRESHOLD) -> Dict:
    """Assign a binary hallucination label to an example in-place and return it.

    Labelling logic differs by condition:
      - unanswerable: label=0 if the model abstained or produced no output;
        label=1 if it generated any non-empty, non-abstention answer.
        token_f1 is set to 0.0 since there is no valid reference.
      - closed_book / answerable: label=0 if token F1 >= f1_threshold, else label=1.

    Args:
        ex: Example dict with keys 'model_answer', 'references', and 'condition'.
        f1_threshold: Minimum token F1 for a non-unanswerable example to be
            considered correct. Defaults to F1_THRESHOLD.

    Returns:
        The same dict with 'token_f1' (float) and 'label' (int, 0=correct / 1=hallucinated) added.
    """
    pred = (ex["model_answer"] or "").strip()
    refs = ex["references"]

    if ex["condition"] == "unanswerable":
        ex["token_f1"] = 0.0
        ex["label"]    = 0 if (len(pred) == 0 or is_abstention(pred)) else 1
    else:
        f1             = token_f1(pred, refs)
        ex["token_f1"] = f1
        ex["label"]    = 0 if f1 >= f1_threshold else 1

    return ex

def build_prompt(ex: Dict) -> str:
    """Construct the input prompt for a given example.

    Prompt format varies by condition:
      - closed_book: no passage, model must answer from memory.
      - answerable: passage contains the answer.
      - unanswerable: passage does not contain the answer, model should abstain.

    Args:
        ex: Example dict with keys 'condition', 'question', and optionally 'context'.

    Returns:
        Prompt string ready for tokenisation.
    """
    if ex["condition"] == "closed_book":
        return (
            "Answer in one word or short phrase only. Do not explain.\n"
            f"Question: {ex['question']}\n"
            "Answer:"
        )
    elif ex["condition"] == "answerable":
        return (
            "Answer the question using the passage below.\n"
            f"Passage: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            "Answer:"
        )
    else:
        return (
            "Answer the question using the passage below\n"
            "If the passage does not contain the answer, "
            "abstain from answering'\n"
            f"Passage: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            "Answer:"
        )

def generate_and_extract(ex: Dict, model, tokenizer, n_layers: int) -> Dict:
    """Run a single greedy generation pass and extract internal features.

    Extracts:
      - model_answer  (str): Decoded answer text.
      - hidden_states (np.ndarray, shape (n_layers, D_MODEL)): Mean-pooled over answer tokens.
      - attn_entropy  (np.ndarray, shape (n_layers,)): Mean attention entropy per layer.
      - logprob_stats (np.ndarray, shape (4,)): [mean, min, max, std] of token log-probs.

    Args:
        ex: Example dict passed to build_prompt.
        model: Loaded HuggingFace causal LM with output_hidden_states=True and output_attentions=True.
        tokenizer: Matching tokenizer.
        n_layers: Number of transformer layers in the model.

    Returns:
        Dict with keys 'model_answer', 'hidden_states', 'attn_entropy', 'logprob_stats'.
    """
    prompt    = build_prompt(ex)
    inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    STOP_TOKENS = [tokenizer.eos_token_id, 198, 271]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            output_attentions=True,
            eos_token_id=STOP_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids   = outputs.sequences[0, prompt_len:]
    model_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    model_answer = model_answer.split("\n")[0].split("Answer:")[0].strip().rstrip(".")

    token_logprobs = []
    for step_scores in outputs.scores:
        logprobs  = F_torch.log_softmax(step_scores[0], dim=-1)
        chosen_id = step_scores[0].argmax()
        token_logprobs.append(logprobs[chosen_id].item())

    logprob_arr   = np.array(token_logprobs)
    logprob_stats = np.array([logprob_arr.mean(), logprob_arr.min(),
                               logprob_arr.max(), logprob_arr.std()])

    layer_vecs = []
    for layer_idx in range(1, n_layers + 1):
        step_vecs = [step_hs[layer_idx][0, -1, :] for step_hs in outputs.hidden_states]
        layer_vecs.append(torch.stack(step_vecs).mean(dim=0))
    hidden_states = torch.stack(layer_vecs).cpu().float().numpy()

    layer_entropies = []
    for layer_idx in range(n_layers):
        step_entropies = []
        for step_attn in outputs.attentions:
            attn      = step_attn[layer_idx][0]
            attn_last = attn[:, -1, :].clamp(min=1e-9)
            entropy   = -(attn_last * attn_last.log()).sum(dim=-1)
            step_entropies.append(entropy.mean().item())
        layer_entropies.append(np.mean(step_entropies))
    attn_entropy = np.array(layer_entropies)

    return {
        "model_answer":  model_answer,
        "hidden_states": hidden_states,
        "attn_entropy":  attn_entropy,
        "logprob_stats": logprob_stats,
    }


def sample_k_answers(ex: Dict, model, tokenizer, k: int = K_SAMPLES) -> List[str]:
    """Generate k stochastic samples for SelfCheckGPT consistency scoring.

    Args:
        ex: Example dict passed to build_prompt.
        model: Loaded HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        k: Number of stochastic samples to generate.

    Returns:
        List of k decoded answer strings.
    """
    prompt    = build_prompt(ex)
    inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    STOP_TOKENS = [tokenizer.eos_token_id, 198, 271]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            num_return_sequences=k,
            eos_token_id=STOP_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        .strip().split("\n")[0].split("Answer:")[0].strip().rstrip(".")
        for seq in outputs
    ] 

def load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features and labels for a given split from saved .npz files.

    Args:
        split_name: One of 'train', 'val', or 'test'.

    Returns:
        Tuple of (hidden_states, attn_entropy, logprob_stats, labels) as float32/int32 arrays.
    """
    npz = np.load(FEATURES_DIR / f"{split_name}_features.npz")
    return (
        npz["hidden_states"].astype(np.float32),
        npz["attn_entropy"].astype(np.float32),
        npz["logprob_stats"].astype(np.float32),
        npz["labels"].astype(np.int32),
    )


def prepare_features(
    hs: np.ndarray,
    ae: np.ndarray,
    lp: np.ndarray,
    hs_scaler: Optional[StandardScaler] = None,
    hs_pca: Optional[PCA] = None,
    ae_scaler: Optional[StandardScaler] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, StandardScaler, PCA, StandardScaler]:
    """Prepare the final feature matrix for classifier training or inference.

    Standardises and PCA-compresses hidden states, standardises attention entropy,
    then concatenates all three feature types.

    Args:
        hs: Hidden states of shape (N, n_layers, D_MODEL).
        ae: Attention entropy of shape (N, n_layers).
        lp: Log-probability statistics of shape (N, 4).
        hs_scaler: Fitted scaler for hidden states. Required if fit=False.
        hs_pca: Fitted PCA for hidden states. Required if fit=False.
        ae_scaler: Fitted scaler for attention entropy. Required if fit=False.
        fit: If True, fit scalers and PCA on the provided data (use for training split only).

    Returns:
        Tuple of (X, hs_scaler, hs_pca, ae_scaler) where X has shape (N, PCA_COMPONENTS + n_layers + 4).
    """
    N      = hs.shape[0]
    hs_flat = hs.reshape(N, -1)

    if fit:
        hs_scaler = StandardScaler()
        hs_pca    = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        hs_pca.fit(hs_scaler.fit_transform(hs_flat))

    hs_out   = hs_pca.transform(hs_scaler.transform(hs_flat))
    ae_clean = np.nan_to_num(ae, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        ae_scaler = StandardScaler()
        ae_scaler.fit(ae_clean)

    ae_out = ae_scaler.transform(ae_clean)
    X      = np.concatenate([hs_out, ae_out, lp], axis=1)

    return X, hs_scaler, hs_pca, ae_scaler


def get_last_layer(
    hs: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    fit: bool = False,
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """Extract and compress the last transformer layer's hidden states.

    Args:
        hs: Hidden states of shape (N, n_layers, D_MODEL).
        scaler: Fitted scaler. Required if fit=False.
        pca: Fitted PCA. Required if fit=False.
        fit: If True, fit scaler and PCA on the provided data.

    Returns:
        Tuple of (features, scaler, pca) where features has shape (N, 64).
    """
    last = hs[:, -1, :]
    if fit:
        scaler = StandardScaler().fit(last)
        pca    = PCA(n_components=64, random_state=SEED)
        pca.fit(scaler.transform(last))
    return pca.transform(scaler.transform(last)), scaler, pca

class MLPProbe(nn.Module):
    """Two-layer MLP probe for binary hallucination classification.

    Architecture: input → hidden_dim → 64 → 1 with ReLU activations and dropout.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    hidden_dim: int = 128, dropout: float = 0.3,
    lr: float = 1e-3, epochs: int = 100, batch_size: int = 64,
    patience: int = 10, device: str = DEVICE,
) -> Tuple[MLPProbe, float]:
    """Train an MLPProbe with early stopping on validation AUC.

    Args:
        X_tr: Training features of shape (N_train, D).
        y_tr: Training binary labels of shape (N_train,).
        X_va: Validation features of shape (N_val, D).
        y_va: Validation binary labels of shape (N_val,).
        hidden_dim: Width of the first hidden layer.
        dropout: Dropout rate applied after each hidden layer.
        lr: AdamW learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early stopping patience (epochs without AUC improvement).
        device: Torch device string.

    Returns:
        Tuple of (trained model with best val AUC weights, best_val_auc).
    """
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)

    loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    model     = MLPProbe(X_tr.shape[1], hidden_dim, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state, no_improve = 0.0, None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        val_auc = roc_auc_score(y_va, va_prob)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict:
    """Compute classification metrics given true labels and predicted probabilities.

    Args:
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities for the positive class, shape (N,).
        name: Identifier string included in the returned dict.

    Returns:
        Dict with keys: name, acc, f1, precision, recall, auc.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "name":      name,
        "acc":       accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }

def compute_scg_scores(examples: List[Dict], scorer) -> np.ndarray:
    """Compute SelfCheckGPT inconsistency scores for a list of examples.

    For each example, compares the greedy model answer against K stochastic
    samples using BERTScore F1. High inconsistency indicates hallucination.

    Args:
        examples: List of example dicts, each containing:
            - 'model_answer' (str): Greedy decoded answer.
            - 'scg_samples' (list[str]): K stochastic samples at temperature > 0.
        scorer: A BERTScorer instance.

    Returns:
        Array of shape (N, 1) with float32 inconsistency scores in [0, 1].
        Returns 0.0 for examples with missing answers or samples.
    """
    scores = []
    for ex in examples:
        greedy  = (ex.get("model_answer") or "").strip()
        samples = [s for s in (ex.get("scg_samples") or []) if s and s.strip()]

        if not greedy or not samples:
            scores.append(0.0)
            continue

        _, _, bert_f1 = scorer.score(cands=[greedy] * len(samples), refs=samples)
        scores.append(1.0 - bert_f1.mean().item())

    return np.array(scores, dtype=np.float32).reshape(-1, 1)


def read_json(filepath) -> List[Dict]:
    """Load a file of JSON objects into a list of dicts.

    Handles two formats produced by the processing pipeline:
      - JSONL: one compact JSON object per line (fast path).
      - Concatenated pretty-printed objects: multi-line objects written
        sequentially without a wrapping array (fallback path).

    Args:
        filepath: Path to the JSON / JSONL file.

    Returns:
        List of parsed dicts in file order.

    Raises:
        json.JSONDecodeError: If the content cannot be parsed by either strategy.
    """
    with open(filepath) as f:
        content = f.read().strip()
    try:
        return [json.loads(l) for l in content.splitlines() if l.strip()]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        objs, idx = [], 0
        while idx < len(content):
            obj, end = decoder.raw_decode(content, idx)
            objs.append(obj)
            idx = end
            while idx < len(content) and content[idx] in " \n\r\t":
                idx += 1
        return objs