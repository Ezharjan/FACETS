"""End-to-end FACETS-PR evaluation pipeline.

For every model and every transcript bundle in the corpus we generate three
drafts (SAE, AAE, and SAE with a red herring); compute the FACETS-PR metrics
on each; compute paired Dialect-Parity values; and emit one CSV row per
(model, item, condition).
"""
from __future__ import annotations

import csv
import gc
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import ollama

from .prompts import (
    render_prompt_draft, render_prompt_review, render_prompt_recall,
)
from .metrics import (
    nehr, vqr, saa, pcs,
    er, ir, ai_idx,
    rhdr, lrs,
    dps,
    dtc, scs,
    iir, pces, emd, bsi,
    frs,
)

DEFAULT_MODELS: tuple[str, ...] = (
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "phi3:mini",
    "gemma2:9b",
)

GEN_OPTIONS = {
    "temperature": 0.2,
    "top_p":       0.9,
    "seed":        0xCDAA48,
    # Cap the response length. Small models sometimes fall into runaway
    # repetition (we saw a >100k-character "draft" out of phi3:mini once);
    # 2048 tokens is about 4x a normal draft, so real reports are unaffected
    # while the pathological cases stay bounded.
    "num_predict": 2048,
    # Pin the context window.  Ollama's per-model default can be as small as
    # 2048 tokens; the drafting prompt plus a long transcript plus the draft
    # can exceed that, which silently truncates the prompt and produces
    # incoherent output.  4096 is supported by all six default models.
    "num_ctx":     4096,
}


# ---------------------------------------------------------------------------
# Ollama wrappers
# ---------------------------------------------------------------------------

def _is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA / GPU *out-of-memory* errors from Ollama.

    Deliberately narrow: a bare "cuda" substring is NOT enough, because many
    fatal driver errors (e.g. PTX JIT failures) also mention CUDA but are not
    memory problems and must not be retried as if they were.
    """
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "out of memory", "oom", "cudamalloc", "cuda malloc",
        "unable to allocate", "failed to allocate", "insufficient memory",
        "kv cache",
    ))


def _is_fatal_gpu_error(exc: BaseException) -> bool:
    """Detect non-recoverable GPU/driver errors (PTX JIT, toolchain, or arch
    mismatch). Retrying these is pointless: they need a driver update or an
    Ollama server restart, not a wait-and-retry loop."""
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "ptx", "jit compilation", "unsupported toolchain",
        "no kernel image", "invalid device function",
        "compiled with an unsupported",
    ))


def _unload_model(model: str) -> None:
    """Ask Ollama to unload a model from VRAM so the next load starts clean."""
    try:
        # keep_alive=0 (int) tells Ollama to unload immediately.
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass  # best-effort; server may itself be in a bad state


def _list_loaded_models() -> list[str]:
    """Return the names of every model currently resident in VRAM."""
    try:
        resp = ollama.ps()
    except Exception:
        return []
    models = getattr(resp, "models", None) or (
        resp.get("models", []) if isinstance(resp, dict) else [])
    names: list[str] = []
    for m in models:
        name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
        if name:
            names.append(name)
    return names


def _unload_all() -> None:
    """Unload *every* resident model from VRAM.

    On OOM the memory pressure usually comes from *other* models that are
    still resident (keep_alive keeps them loaded), not from the model we are
    trying to load — so unloading only the current model is a no-op.  This
    frees the whole GPU before we retry.
    """
    for name in _list_loaded_models():
        _unload_model(name)


def _ollama_generate(model: str, prompt: str, options: dict | None = None,
                     keep_alive: str = "30m", retries: int = 3) -> str:
    opts = dict(GEN_OPTIONS)
    if options:
        opts.update(options)
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            res = ollama.generate(model=model, prompt=prompt, options=opts,
                                  keep_alive=keep_alive)
            # Handle both dict and object responses from ollama SDK.
            if isinstance(res, dict):
                return (res.get("response", "") or "").strip()
            return (getattr(res, "response", "") or "").strip()
        except Exception as e:
            last_err = e
            # Fatal GPU/driver errors (PTX JIT, toolchain/arch mismatch) won't
            # recover by waiting — fail fast with an actionable message instead
            # of burning through every retry.
            if _is_fatal_gpu_error(e):
                raise RuntimeError(
                    f"Non-recoverable GPU/driver error for {model}: {e}\n"
                    "This is a PTX/CUDA-toolchain problem, NOT out-of-memory. "
                    "Restart the Ollama server to clear a stale GPU context, "
                    "and if it persists update your NVIDIA driver (then reboot). "
                    "Retrying in-process will not help."
                ) from e
            # Don't sleep after the final failed attempt.
            if attempt == retries:
                break
            if _is_oom_error(e):
                # OOM: free *all* resident models (the pressure is usually from
                # other models still loaded under keep_alive), GC, wait, retry.
                print(f"[pipeline] CUDA OOM on attempt {attempt+1}/{retries+1} "
                      f"for {model}, unloading all models and waiting …",
                      flush=True)
                _unload_all()
                gc.collect()
                time.sleep(10 * (attempt + 1))   # 10s, 20s, 30s
            else:
                # Transient / network error: short backoff.
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama generate failed for {model}: {last_err}")


def _ensure_models(models: Iterable[str]) -> None:
    """Confirm each model is locally available; if not, pull it."""
    resp = ollama.list()
    # ollama SDK >=0.3 returns a ListResponse object; handle both dict and object.
    models_list = getattr(resp, "models", None) or (resp.get("models", []) if isinstance(resp, dict) else [])
    available = set()
    for m in models_list:
        name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
        if name:
            available.add(name)
    for m in models:
        if m in available or m.replace(":", "_") in available:
            continue
        print(f"[pipeline] pulling {m} ...", flush=True)
        for _ in ollama.pull(m, stream=True):
            pass


# ---------------------------------------------------------------------------
# Per-item computation
# ---------------------------------------------------------------------------

def _attach_turns(annotations: dict, item: dict) -> dict:
    """Add the turns list as ``_turns`` to the annotation dict (used by metric
    helpers that need turn-level access)."""
    out = dict(annotations)
    out["_turns"]   = item["turns"]
    return out


def _compute_axis_metrics(report: str, item: dict,
                          transcript_text: str) -> dict:
    """Compute every axis metric except RHDR (which needs the reviewed draft)
    and DPS (cross-variant)."""
    ann = _attach_turns(item["annotations"], item)
    F = {
        "NEHR": nehr(report, ann, transcript_text),
        "VQR":  vqr(report, ann),
        "SAA":  saa(report, ann),
        "PCS":  pcs(report),
    }
    A = {
        "ER":  er(report, ann),
        "IR":  ir(report, ann),
        "AI":  ai_idx(report, ann),
    }
    T = {
        "DTC": dtc(report),
        "SCS": scs(report, transcript_text),
    }
    S = {
        "IIR":  iir(report, transcript_text),
        "PCES": pces(report),
        "EMD":  emd(report, transcript_text),
        "BSI":  bsi(report),
    }
    return {"F": F, "A": A, "T": T, "S": S}


def _ground_truth_facts(item: dict) -> list[str]:
    """Return the inculpatory + exculpatory facts the LRS prompt should
    expect a memoryless reviewer to recall."""
    turns = item["turns"]
    idx = (item["annotations"].get("inculpatory_indices") or []) \
        + (item["annotations"].get("exculpatory_indices") or [])
    return [turns[i]["text"] for i in idx if i < len(turns)]


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    corpus_path: str
    out_csv:     str
    models:      tuple[str, ...] = DEFAULT_MODELS
    do_recall:   bool = False     # C2 LRS is optional / expensive
    ensure_pull: bool = True
    keep_alive:  str  = "30m"
    write_drafts_to: str | None = None  # if set, also dump per-item drafts
    resume:      bool = False     # skip (model, item, variant) already in CSV


def _csv_columns() -> list[str]:
    return [
        "model", "item_id", "category", "condition", "variant",
        "NEHR", "VQR", "SAA", "PCS",
        "ER", "IR", "AI",
        "RHDR", "LRS",
        "DTC", "SCS",
        "IIR", "PCES", "EMD", "BSI",
        "FRS",
        "DPS_NEHR", "DPS_VQR", "DPS_SAA", "DPS_PCS",
        "DPS_ER", "DPS_IR", "DPS_AI",
        "DPS_DTC", "DPS_SCS",
        "DPS_IIR", "DPS_PCES", "DPS_EMD", "DPS_BSI",
    ]


def _row_from(model: str, bundle: dict, variant: str, item: dict,
              axis_metrics: dict, rhdr_v: float | None, lrs_v: float | None,
              dps_map: dict[str, float] | None) -> dict:
    row = {
        "model":     model,
        "item_id":   bundle["id"],
        "category":  bundle["category"],
        "condition": bundle["condition"],
        "variant":   variant,
        **axis_metrics["F"],
        **axis_metrics["A"],
        "RHDR": rhdr_v,
        "LRS":  lrs_v,
        **axis_metrics["T"],
        **axis_metrics["S"],
    }
    # FRS: include Axis E only on AAE rows (where DPS exists).
    per_axis = {**axis_metrics}
    if dps_map:
        per_axis["E"] = dps_map
    if rhdr_v is not None or lrs_v is not None:
        per_axis["C"] = {"RHDR": rhdr_v, "LRS": lrs_v}
    row["FRS"] = frs(per_axis)
    if dps_map:
        for k, v in dps_map.items():
            row[f"DPS_{k}"] = v
    return row


_ALL_VARIANTS = frozenset(("sae", "aae", "sae_rh"))


def _load_done_keys(csv_path: str) -> set[tuple[str, str, str]]:
    """Return {(model, item_id, variant)} for *complete* bundles in *csv_path*.

    A bundle is complete when all three variants (sae, aae, sae_rh) are
    present for the same (model, item_id).  Partial bundles are **removed**
    from the CSV on disk so they don't produce duplicates when we re-run.
    """
    if not os.path.exists(csv_path):
        return set()

    # -- pass 1: read all rows, group by (model, item_id) --
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate that the header matches what we expect.
        expected = _csv_columns()
        if reader.fieldnames is not None and list(reader.fieldnames) != expected:
            raise RuntimeError(
                f"[resume] CSV header mismatch in {csv_path}.\n"
                f"  expected: {expected[:5]}…\n"
                f"  got:      {list(reader.fieldnames)[:5]}…\n"
                "Delete the file or run without --resume."
            )
        all_rows: list[dict] = list(reader)

    # -- identify complete vs partial bundles --
    from collections import defaultdict
    bundles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in all_rows:
        key = (row["model"], row["item_id"])
        bundles[key].add(row["variant"])

    complete = {k for k, variants in bundles.items()
                if variants >= _ALL_VARIANTS}

    # -- pass 2: rewrite CSV keeping only complete-bundle rows --
    has_partial = any(variants and variants < _ALL_VARIANTS
                      for variants in bundles.values())
    if has_partial:
        kept = [r for r in all_rows
                if (r["model"], r["item_id"]) in complete]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=expected,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
        n_dropped = len(all_rows) - len(kept)
        print(f"[pipeline] resume: dropped {n_dropped} rows from "
              f"incomplete bundles", flush=True)

    done: set[tuple[str, str, str]] = set()
    for k in complete:
        for v in _ALL_VARIANTS:
            done.add((*k, v))
    return done


def run(cfg: RunConfig) -> str:
    """Run the full pipeline and write ``cfg.out_csv``.  Returns the path."""
    from .corpus.generator import load_corpus

    corpus = load_corpus(cfg.corpus_path)
    print(f"[pipeline] loaded {len(corpus)} bundles from {cfg.corpus_path}")

    if cfg.ensure_pull:
        _ensure_models(cfg.models)

    # ------------------------------------------------------------------
    # Resume support: read existing rows so we can skip them.
    # ------------------------------------------------------------------
    done_keys: set[tuple[str, str, str]] = set()
    if cfg.resume and os.path.exists(cfg.out_csv):
        done_keys = _load_done_keys(cfg.out_csv)
        print(f"[pipeline] resuming – {len(done_keys)} rows already done")

    os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
    file_mode = "a" if cfg.resume and done_keys else "w"
    fh = open(cfg.out_csv, file_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_csv_columns(), extrasaction="ignore")
    if file_mode == "w":
        writer.writeheader()

    drafts_fh = None
    if cfg.write_drafts_to:
        os.makedirs(os.path.dirname(cfg.write_drafts_to) or ".", exist_ok=True)
        drafts_mode = "a" if cfg.resume and done_keys else "w"
        drafts_fh = open(cfg.write_drafts_to, drafts_mode, encoding="utf-8")

    total = len(cfg.models) * len(corpus) * 3
    counter = 0
    skipped = 0
    t0 = time.time()
    try:
        for model in cfg.models:
            # Free VRAM held by any previously-loaded model so the new model
            # (and its KV cache) has room.  Without this, keep_alive keeps each
            # finished model resident and successive loads eventually OOM.
            _unload_all()
            gc.collect()

            for bundle in corpus:
                bid = bundle["id"]

                # --- Check resume: skip whole bundle if all 3 variants done ---
                if done_keys and all(
                    (model, bid, v) in done_keys for v in ("sae", "aae", "sae_rh")
                ):
                    counter += 3
                    skipped += 3
                    continue

                sae_item = bundle["sae"]
                aae_item = bundle["aae"]
                rh_item  = bundle["sae_rh"]
                sae_txt  = sae_item["transcript_text"]
                aae_txt  = aae_item["transcript_text"]
                rh_txt   = rh_item["transcript_text"]

                # Draft -- SAE
                counter += 1
                print(f"[{counter:>4}/{total}] {model} :: {bid} :: sae",
                      flush=True)
                R_sae = _ollama_generate(model,
                                         render_prompt_draft(sae_txt, bundle["category"]),
                                         keep_alive=cfg.keep_alive)
                sae_metrics = _compute_axis_metrics(R_sae, sae_item, sae_txt)
                lrs_v = None
                if cfg.do_recall:
                    facts = _ground_truth_facts(sae_item)
                    recall_out = _ollama_generate(
                        model,
                        render_prompt_recall(bundle["category"]),
                        keep_alive=cfg.keep_alive,
                    )
                    lrs_v = lrs(recall_out, facts)

                # Draft -- AAE
                counter += 1
                print(f"[{counter:>4}/{total}] {model} :: {bid} :: aae",
                      flush=True)
                R_aae = _ollama_generate(model,
                                         render_prompt_draft(aae_txt, bundle["category"]),
                                         keep_alive=cfg.keep_alive)
                aae_metrics = _compute_axis_metrics(R_aae, aae_item, aae_txt)

                # Build DPS map (E1) for the AAE row.
                dps_map = {
                    "NEHR": dps(sae_metrics["F"]["NEHR"], aae_metrics["F"]["NEHR"]),
                    "VQR":  dps(sae_metrics["F"]["VQR"],  aae_metrics["F"]["VQR"]),
                    "SAA":  dps(sae_metrics["F"]["SAA"],  aae_metrics["F"]["SAA"]),
                    "PCS":  dps(sae_metrics["F"]["PCS"],  aae_metrics["F"]["PCS"]),
                    "ER":   dps(sae_metrics["A"]["ER"],   aae_metrics["A"]["ER"]),
                    "IR":   dps(sae_metrics["A"]["IR"],   aae_metrics["A"]["IR"]),
                    "AI":   dps(sae_metrics["A"]["AI"],   aae_metrics["A"]["AI"]),
                    "DTC":  dps(sae_metrics["T"]["DTC"],  aae_metrics["T"]["DTC"]),
                    "SCS":  dps(sae_metrics["T"]["SCS"],  aae_metrics["T"]["SCS"]),
                    "IIR":  dps(sae_metrics["S"]["IIR"],  aae_metrics["S"]["IIR"]),
                    "PCES": dps(sae_metrics["S"]["PCES"], aae_metrics["S"]["PCES"]),
                    "EMD":  dps(sae_metrics["S"]["EMD"],  aae_metrics["S"]["EMD"]),
                    "BSI":  dps(sae_metrics["S"]["BSI"],  aae_metrics["S"]["BSI"]),
                }

                writer.writerow(_row_from(model, bundle, "sae", sae_item,
                                          sae_metrics, None, lrs_v, None))
                writer.writerow(_row_from(model, bundle, "aae", aae_item,
                                          aae_metrics, None, None, dps_map))

                # Draft + Review -- SAE+RH for Axis C, C1
                counter += 1
                print(f"[{counter:>4}/{total}] {model} :: {bid} :: sae_rh",
                      flush=True)
                R_rh = _ollama_generate(model,
                                        render_prompt_draft(rh_txt, bundle["category"]),
                                        keep_alive=cfg.keep_alive)
                R_rh_rev = _ollama_generate(model,
                                            render_prompt_review(R_rh),
                                            keep_alive=cfg.keep_alive)
                rh_metrics = _compute_axis_metrics(R_rh_rev, rh_item, rh_txt)
                rhdr_v = rhdr(R_rh_rev, rh_item["annotations"])
                writer.writerow(_row_from(model, bundle, "sae_rh", rh_item,
                                          rh_metrics, rhdr_v, None, None))
                fh.flush()

                if drafts_fh is not None:
                    json.dump({
                        "model":  model,
                        "item":   bid,
                        "drafts": {
                            "sae":      R_sae,
                            "aae":      R_aae,
                            "sae_rh":   R_rh,
                            "sae_rh_reviewed": R_rh_rev,
                        },
                    }, drafts_fh)
                    drafts_fh.write("\n")
                    drafts_fh.flush()

                # Periodic GC to curb host-side memory growth.
                gc.collect()

    finally:
        fh.close()
        if drafts_fh is not None:
            drafts_fh.close()

    elapsed = time.time() - t0
    print(f"[pipeline] wrote {cfg.out_csv} in {elapsed:.1f}s "
          f"({skipped} skipped via resume)")
    return cfg.out_csv
