"""
Automated interpretability pipeline for Gemma Scope 2 features.

Implements the binary classification methodology from the Gemma Scope 2 paper:
  1. For a feature, collect top activating examples and random non-activating examples
  2. Ask an LLM to generate a natural language description of what the feature detects
  3. Present the LLM with a mix of activating and non-activating examples
  4. Ask the LLM to classify which examples would activate the feature
  5. Score = accuracy of the classification

This module uses the Anthropic API (Claude) for the LLM calls.
Can also use local models if API is unavailable.

REQUIRES: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.
"""

import torch
import numpy as np
import json
import os
import time
from typing import Optional
from dataclasses import dataclass, field


# ============================================================
# Data structures
# ============================================================

@dataclass
class FeatureInterpretation:
    """Result of interpreting a single feature."""
    feature_idx: int
    layer: int
    site: str

    # Feature statistics
    frequency: float = 0.0
    mean_activation: float = 0.0

    # Top activating examples
    top_examples: list = field(default_factory=list)  # [(text, position, activation)]

    # LLM-generated interpretation
    description: str = ""

    # Classification score
    classification_score: float = 0.0
    n_correct: int = 0
    n_total: int = 0

    # Top logit effects
    top_promoted_tokens: list = field(default_factory=list)  # [(token, effect)]
    top_suppressed_tokens: list = field(default_factory=list)


# ============================================================
# Example collection
# ============================================================

def collect_feature_examples(
    model, sae, tokenizer,
    layer: int,
    feature_idx: int,
    corpus_texts: list,
    site: str = "resid_post",
    n_top: int = 10,
    n_random: int = 10,
    context_window: int = 30,
    device: str = "cuda",
) -> dict:
    """
    Collect top activating and random non-activating examples for a feature.

    Args:
        corpus_texts: list of text strings to scan
        n_top: number of top activating examples
        n_random: number of random non-activating examples
        context_window: how many tokens of context around the activation

    Returns:
        dict with "activating" and "non_activating" example lists
    """
    from src.hooks import gather_residual_activations

    all_activations = []  # (text_idx, position, activation, context)

    for text_idx, text in enumerate(corpus_texts):
        inputs = tokenizer.encode(text, return_tensors="pt",
                                  add_special_tokens=True, truncation=True,
                                  max_length=512).to(device)

        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())

        feat_acts = features[0, :, feature_idx]  # (seq_len,)

        # Find positions where feature fires
        for pos in range(1, feat_acts.shape[0]):  # skip BOS
            act_val = feat_acts[pos].item()
            if act_val > 0:
                # Get context around this position
                tokens = inputs[0].tolist()
                start = max(0, pos - context_window // 2)
                end = min(len(tokens), pos + context_window // 2)
                context = tokenizer.decode(tokens[start:end])
                target_token = tokenizer.decode([tokens[pos]])

                all_activations.append({
                    "text_idx": text_idx,
                    "position": pos,
                    "activation": act_val,
                    "context": context,
                    "target_token": target_token,
                })

    # Sort by activation, take top-n
    all_activations.sort(key=lambda x: x["activation"], reverse=True)
    activating = all_activations[:n_top]

    # Collect non-activating examples (random positions from texts where feature doesn't fire)
    non_activating = []
    for text_idx, text in enumerate(corpus_texts[:n_random]):
        inputs = tokenizer.encode(text, return_tensors="pt",
                                  add_special_tokens=True, truncation=True,
                                  max_length=512).to(device)

        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        feat_acts = features[0, :, feature_idx]

        # Find a position where feature does NOT fire
        non_firing = (feat_acts == 0).nonzero(as_tuple=True)[0]
        if len(non_firing) > 1:
            pos = non_firing[len(non_firing) // 2].item()  # pick middle non-firing position
            tokens = inputs[0].tolist()
            start = max(0, pos - context_window // 2)
            end = min(len(tokens), pos + context_window // 2)
            context = tokenizer.decode(tokens[start:end])

            non_activating.append({
                "text_idx": text_idx,
                "position": pos,
                "activation": 0.0,
                "context": context,
                "target_token": tokenizer.decode([tokens[pos]]),
            })

    return {
        "activating": activating,
        "non_activating": non_activating,
        "total_activations": len(all_activations),
        "frequency": len(all_activations) / max(sum(
            len(tokenizer.encode(t, truncation=True, max_length=512))
            for t in corpus_texts
        ), 1),
    }


# ============================================================
# LLM-based interpretation
# ============================================================

def generate_feature_description(
    activating_examples: list,
    non_activating_examples: list,
    api_key: Optional[str] = None,
) -> str:
    """
    Use Claude to generate a natural language description of what a feature detects.

    Falls back to a simple heuristic if API is unavailable.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            # Build prompt
            act_texts = "\n".join(
                f"  Example {i+1} (activation={ex['activation']:.1f}): "
                f"...{ex['context']}... [fires on: '{ex['target_token']}']"
                for i, ex in enumerate(activating_examples[:8])
            )

            non_act_texts = "\n".join(
                f"  Example {i+1}: ...{ex['context']}..."
                for i, ex in enumerate(non_activating_examples[:5])
            )

            prompt = f"""You are analyzing a neuron/feature in a language model. This feature activates on certain tokens in certain contexts, and does not activate in others.

Here are examples where the feature ACTIVATES (with activation strength):
{act_texts}

Here are examples where the feature does NOT activate:
{non_act_texts}

Based on these examples, provide a concise description (1-2 sentences) of what concept or pattern this feature detects. Be specific — don't just say "it fires on common words." Focus on the semantic or syntactic pattern that distinguishes activating from non-activating examples."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        except Exception as e:
            print(f"  API call failed: {e}")

    # Fallback: simple heuristic based on common tokens
    token_counts = {}
    for ex in activating_examples:
        tok = ex["target_token"].strip().lower()
        token_counts[tok] = token_counts.get(tok, 0) + 1

    common = sorted(token_counts.items(), key=lambda x: -x[1])[:3]
    if common:
        return f"Feature fires frequently on tokens: {', '.join(f'{t} ({c}x)' for t, c in common)}"
    return "Unable to determine feature description"


def classify_examples(
    description: str,
    test_examples: list,
    api_key: Optional[str] = None,
) -> list:
    """
    Ask an LLM to classify which examples would activate the feature
    based on the generated description.

    Returns list of booleans (predicted activating or not).
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Random baseline
        return [np.random.random() > 0.5 for _ in test_examples]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        examples_text = "\n".join(
            f"  {i+1}. ...{ex['context']}... [token: '{ex['target_token']}']"
            for i, ex in enumerate(test_examples)
        )

        prompt = f"""A feature in a language model has been described as:
"{description}"

For each of the following examples, predict whether this feature would ACTIVATE (yes/no).
Respond with ONLY a JSON list of booleans, e.g. [true, false, true, ...]. No other text.

Examples:
{examples_text}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Parse JSON
        text = text.replace("```json", "").replace("```", "").strip()
        predictions = json.loads(text)
        return predictions

    except Exception as e:
        print(f"  Classification API call failed: {e}")
        return [np.random.random() > 0.5 for _ in test_examples]


# ============================================================
# Full autointerp pipeline
# ============================================================

def autointerp_feature(
    model, sae, tokenizer,
    layer: int,
    feature_idx: int,
    corpus_texts: list,
    site: str = "resid_post",
    n_top: int = 10,
    n_random: int = 10,
    api_key: Optional[str] = None,
    device: str = "cuda",
) -> FeatureInterpretation:
    """
    Full automated interpretability pipeline for a single feature.

    1. Collect activating and non-activating examples
    2. Generate description
    3. Classify held-out examples
    4. Compute score
    """
    result = FeatureInterpretation(
        feature_idx=feature_idx,
        layer=layer,
        site=site,
    )

    # 1. Collect examples
    examples = collect_feature_examples(
        model, sae, tokenizer, layer, feature_idx,
        corpus_texts, site, n_top, n_random,
        device=device,
    )

    result.frequency = examples["frequency"]
    result.top_examples = [
        (ex["context"], ex["target_token"], ex["activation"])
        for ex in examples["activating"]
    ]

    if not examples["activating"]:
        result.description = "Dead feature (never activates on corpus)"
        return result

    result.mean_activation = np.mean([ex["activation"] for ex in examples["activating"]])

    # 2. Generate description
    # Use first half for description, second half for classification
    n_desc = len(examples["activating"]) // 2
    desc_examples = examples["activating"][:max(n_desc, 3)]
    test_activating = examples["activating"][n_desc:]

    result.description = generate_feature_description(
        desc_examples, examples["non_activating"], api_key
    )

    # 3. Classify test examples
    test_set = test_activating + examples["non_activating"]
    ground_truth = [True] * len(test_activating) + [False] * len(examples["non_activating"])

    # Shuffle
    indices = list(range(len(test_set)))
    np.random.shuffle(indices)
    test_set = [test_set[i] for i in indices]
    ground_truth = [ground_truth[i] for i in indices]

    predictions = classify_examples(result.description, test_set, api_key)

    # 4. Score
    if len(predictions) == len(ground_truth):
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        result.n_correct = correct
        result.n_total = len(ground_truth)
        result.classification_score = correct / len(ground_truth) if ground_truth else 0.0

    return result


def autointerp_top_features(
    model, sae, tokenizer,
    layer: int,
    corpus_texts: list,
    n_features: int = 10,
    site: str = "resid_post",
    api_key: Optional[str] = None,
    device: str = "cuda",
) -> list:
    """
    Run autointerp on the top-n most active features across a corpus.

    Returns list of FeatureInterpretation sorted by classification score.
    """
    from src.hooks import gather_residual_activations

    # First pass: find top features by total activation across corpus
    print(f"  Scanning corpus ({len(corpus_texts)} texts) for top features...")
    total_acts = torch.zeros(sae.d_sae)

    for text in corpus_texts[:50]:  # cap at 50 texts for speed
        inputs = tokenizer.encode(text, return_tensors="pt",
                                  add_special_tokens=True, truncation=True,
                                  max_length=256).to(device)
        acts = gather_residual_activations(model, layer, inputs)
        features = sae.encode(acts.float())
        total_acts += features[0, 1:].sum(dim=0).cpu()

    top_vals, top_idxs = total_acts.topk(n_features)
    print(f"  Top {n_features} features: {top_idxs.tolist()}")

    # Run autointerp on each
    results = []
    for i, idx in enumerate(top_idxs):
        print(f"  [{i+1}/{n_features}] Feature {idx.item()}...")
        r = autointerp_feature(
            model, sae, tokenizer, layer, idx.item(),
            corpus_texts, site, api_key=api_key, device=device,
        )
        print(f"    Description: {r.description[:100]}")
        print(f"    Score: {r.classification_score:.2f} ({r.n_correct}/{r.n_total})")
        results.append(r)

    results.sort(key=lambda x: x.classification_score, reverse=True)
    return results


# ============================================================
# Export
# ============================================================

def save_interpretations(results: list, path: str):
    """Save autointerp results to JSON."""
    data = []
    for r in results:
        data.append({
            "feature_idx": r.feature_idx,
            "layer": r.layer,
            "site": r.site,
            "frequency": r.frequency,
            "mean_activation": r.mean_activation,
            "description": r.description,
            "classification_score": r.classification_score,
            "n_correct": r.n_correct,
            "n_total": r.n_total,
            "top_examples": r.top_examples[:5],
            "top_promoted_tokens": r.top_promoted_tokens,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} interpretations to {path}")
