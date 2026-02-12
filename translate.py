# translate.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional

from docx import Document
from langgraph.graph import StateGraph, END
from openai import OpenAI

from chunking import (
    Block,
    BlockType,
    iter_blocks,
    chunk_blocks_with_spans,
    build_contexts,
)
from prompts import PROMPTS

# ============================================================
# 0) Types
# ============================================================

Section = Literal["default", "claims", "abstract"]

class TranslateState(TypedDict):
    chunks: List[List[Block]]
    contexts: List[Dict[str, str]]
    i: int
    results: Dict[str, str]

    model: str
    base_url: str | None
    api_key: str | None

    target_lang: str

    # routing / prompts
    section: Section
    prompt_default: str
    prompt_claims: str
    prompt_abstract: str

    # abstract word-count tracking (NEW)
    abstract_word_count: int
    last_abstract_block_id: str | None


# ============================================================
# 1) OpenAI client + JSON extraction fallback
# ============================================================

def make_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    kwargs = {"timeout": 120.0, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def extract_first_json_object(s: str) -> Optional[str]:
    s2 = (s or "").strip()
    if s2.startswith("{") and s2.endswith("}"):
        return s2

    start = s2.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s2)):
        ch = s2[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s2[start : i + 1]
    return None


# ============================================================
# 2) Translate one chunk with a given prompt
# ============================================================

def translate_chunk(
    client: OpenAI,
    model: str,
    *,
    prompt_name: str,
    target_lang: str,
    chunk: List[Block],
    context_before: str = "",
    context_after: str = "",
) -> Dict[str, str]:
    prompt = PROMPTS[prompt_name]
    payload = [{"id": b.id, "text": b.text} for b in chunk]

    system_prompt = prompt.system
    user_prompt = prompt.render_user(
        payload=payload,
        target_lang=target_lang,
        context_before=context_before,
        context_after=context_after,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.0,
    )

    content = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(content)
    except Exception:
        json_str = extract_first_json_object(content)
        if not json_str:
            raise ValueError(f"Model did not return JSON. Got:\n{content[:800]}")
        data = json.loads(json_str)

    out: Dict[str, str] = {}
    translations = data.get("translations", [])
    if isinstance(translations, list):
        for item in translations:
            if isinstance(item, dict) and "id" in item and "text" in item:
                out[str(item["id"])] = str(item["text"])
    return out


# ============================================================
# 3) Abstract word counting + finalization (NEW)
# ============================================================

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def count_english_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))

def append_word_count_to_last_sentence(text: str, n: int) -> str:
    s = (text or "").rstrip()
    if not s:
        return s
    if re.search(r"\(\d+\s+words\)\s*$", s):
        return s
    return f"{s} ({n} words)"


# ============================================================
# 4) Section detection + routing (UPDATED headings)
# ============================================================

def _chunk_text(chunk: List[Block]) -> str:
    return "\n".join((b.text or "") for b in chunk)

def detect_section_from_chunk(prev: Section, chunk: List[Block]) -> Section:
    """
    Claims starts with 【청구 범위】 (sometimes without space).
    Abstract starts with 【요약】.
    Sticky behavior: if no heading is found, keep previous section.
    """
    t = _chunk_text(chunk)

    if "【청구" in t :
        return "claims"

    if "【요약】" in t:
        return "abstract"

    return prev

def node_route_section(state: TranslateState) -> TranslateState:
    i = state["i"]

    # If we are done, finalize abstract once here
    if i >= len(state["chunks"]):
        if state.get("section") == "abstract":
            n = int(state.get("abstract_word_count", 0))
            last_id = state.get("last_abstract_block_id")
            if n > 0 and last_id and last_id in state["results"]:
                results = dict(state["results"])
                results[last_id] = append_word_count_to_last_sentence(results[last_id], n)
                return {**state, "results": results}
        return state

    prev = state.get("section", "default")
    new = detect_section_from_chunk(prev, state["chunks"][i])

    # Leaving abstract -> finalize "(XXX words)" on last abstract sentence
    if prev == "abstract" and new != "abstract":
        n = int(state.get("abstract_word_count", 0))
        last_id = state.get("last_abstract_block_id")
        if last_id and last_id in state["results"]:
            results = dict(state["results"])
            results[last_id] = append_word_count_to_last_sentence(results[last_id], n)
            print(f"[ABSTRACT] finalize word count: {n} words")
            state = {**state, "results": results}

    if new != prev:
        print(f"[ROUTE] chunk {i}: {prev} -> {new}")

    return {**state, "section": new}


def route_after_router(state: TranslateState) -> str:
    if state["i"] >= len(state["chunks"]):
        return END

    sec = state.get("section", "default")
    if sec == "claims":
        return "translate_claims"
    if sec == "abstract":
        return "translate_abstract"
    return "translate_default"


# ============================================================
# 5) Translation nodes (default/claims/abstract)
# ============================================================

def _translate_with_prompt(state: TranslateState, prompt_name: str) -> TranslateState:
    i = state["i"]
    chunks = state["chunks"]
    contexts = state.get("contexts", [])
    if i >= len(chunks):
        return state

    client = make_client(state.get("base_url"), state.get("api_key"))
    chunk = chunks[i]
    ctx_before = contexts[i].get("before", "") if i < len(contexts) else ""
    ctx_after = contexts[i].get("after", "") if i < len(contexts) else ""
    sec = state.get("section", "default")

    print(f"[{sec}] {i}-th chunk...")

    translated_map = translate_chunk(
        client=client,
        model=state["model"],
        prompt_name=prompt_name,
        target_lang=state["target_lang"],
        chunk=chunk,
        context_before=ctx_before,
        context_after=ctx_after,
    )

    results = dict(state["results"])
    results.update(translated_map)

    # Track abstract word count
    abstract_word_count = int(state.get("abstract_word_count", 0))
    last_abstract_block_id = state.get("last_abstract_block_id")

    if sec == "abstract":
        for bid, txt in translated_map.items():
            t = (txt or "").strip()
            # skip heading-only
            if t.upper() == "ABSTRACT":
                continue
            abstract_word_count += count_english_words(t)
            last_abstract_block_id = bid


    return {
        **state,
        "results": results,
        "i": i + 1,
        "abstract_word_count": abstract_word_count,
        "last_abstract_block_id": last_abstract_block_id,
    }


def node_translate_default(state: TranslateState) -> TranslateState:
    return _translate_with_prompt(state, state["prompt_default"])

def node_translate_claims(state: TranslateState) -> TranslateState:
    return _translate_with_prompt(state, state["prompt_claims"])

def node_translate_abstract(state: TranslateState) -> TranslateState:
    return _translate_with_prompt(state, state["prompt_abstract"])


# ============================================================
# 6) Build graph (router -> specialized translate -> router ...)
# ============================================================

def build_translation_graph():
    g = StateGraph(TranslateState)

    g.add_node("route_section", node_route_section)
    g.add_node("translate_default", node_translate_default)
    g.add_node("translate_claims", node_translate_claims)
    g.add_node("translate_abstract", node_translate_abstract)

    g.set_entry_point("route_section")

    g.add_conditional_edges(
        "route_section",
        route_after_router,
        {
            "translate_default": "translate_default",
            "translate_claims": "translate_claims",
            "translate_abstract": "translate_abstract",
            END: END,
        },
    )

    g.add_edge("translate_default", "route_section")
    g.add_edge("translate_claims", "route_section")
    g.add_edge("translate_abstract", "route_section")

    return g.compile()


# ============================================================
# 7) Apply translations back into DOCX
# ============================================================

def apply_translations_to_docx(
    src_docx_path: str | Path,
    translations: Dict[str, str],
    out_docx_path: str | Path,
) -> None:
    doc = Document(str(src_docx_path))

    for block_id, translated in translations.items():
        if block_id.startswith("p:"):
            _, idx = block_id.split(":", 1)
            pi = int(idx)
            if 0 <= pi < len(doc.paragraphs):
                doc.paragraphs[pi].text = translated

        elif block_id.startswith("cell:"):
            parts = block_id.split(":")
            if len(parts) == 5:
                _, ti, ri, ci, pi = parts
                ti = int(ti); ri = int(ri); ci = int(ci); pi = int(pi)

                if 0 <= ti < len(doc.tables):
                    table = doc.tables[ti]
                    if 0 <= ri < len(table.rows):
                        row = table.rows[ri]
                        if 0 <= ci < len(row.cells):
                            cell = row.cells[ci]
                            if 0 <= pi < len(cell.paragraphs):
                                cell.paragraphs[pi].text = translated

    doc.save(str(out_docx_path))


# ============================================================
# 8) End-to-end runner
# ============================================================

def translate_docx(
    src_docx_path: str | Path,
    out_docx_path: str | Path,
    *,
    model: str,
    target_lang: str = "English",
    max_chars_per_chunk: int = 2500,
    context_chars: int = 400,
    chunk_overlap: int = 0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    prompt_default: str = "patent_kr2en_body_v1",
    prompt_claims: str = "patent_kr2en_claims_v1",
    prompt_abstract: str = "patent_kr2en_abstract_v1",
) -> None:
    print("Start chunking...")
    blocks = iter_blocks(src_docx_path)
    chunk_spans = chunk_blocks_with_spans(
        blocks,
        max_chars=max_chars_per_chunk,
        overlap=chunk_overlap,
    )
    chunks = [c for (c, _, _) in chunk_spans]
    contexts_raw = build_contexts(
        blocks,
        [(s, e) for (_, s, e) in chunk_spans],
        context_chars=context_chars,
    )
    contexts = [{"before": b, "after": a} for (b, a) in contexts_raw]
    print(f"Finish chunking: {len(chunks)} chunks (context ±{context_chars} chars)")

    print("Build graph...")
    app = build_translation_graph()

    print("Start invoking...")
    final = app.invoke(
        {
            "chunks": chunks,
            "contexts": contexts,
            "i": 0,
            "results": {},
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "target_lang": target_lang,

            "section": "default",
            "prompt_default": prompt_default,
            "prompt_claims": prompt_claims,
            "prompt_abstract": prompt_abstract,

            # NEW init
            "abstract_word_count": 0,
            "last_abstract_block_id": None,
        },
        config={"recursion_limit": max(50, len(chunks) * 3)},
    )

    # If the document ends while still in abstract, finalize too
    if final.get("section") == "abstract":
        # n = int(final.get("abstract_word_count", 0))
        # last_id = final.get("last_abstract_block_id")
        # if last_id and last_id in final["results"]:
        #     final["results"][last_id] = append_word_count_to_last_sentence(final["results"][last_id], n)
        #     print(f"[ABSTRACT] finalize word count at end: {n} words")

        # Always finalize if we ever saw abstract blocks
        n = int(final.get("abstract_word_count", 0))
        last_id = final.get("last_abstract_block_id")

        if n > 0 and last_id and last_id in final["results"]:
            final["results"][last_id] = append_word_count_to_last_sentence(final["results"][last_id], n)
            print(f"[ABSTRACT] finalize word count: {n} words")


    print("Apply translations...")
    apply_translations_to_docx(src_docx_path, final["results"], out_docx_path)
    print("Done.")
