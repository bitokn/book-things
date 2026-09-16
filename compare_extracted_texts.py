#!/usr/bin/env python3
"""
compare_extracted_texts.py

Compares two .txt files extracted from PDFs of the same source text.
Categorizes differences into:
  1. MAJOR STRUCTURAL CHANGES: Multi-word block additions, deletions, or paragraph rewrites (e.g. >= 10 words).
  2. VOCABULARY EDITS: Minor 1-to-few word substitutions or rewordings ("outmaneuver" <-> "outsmart").
  3. PUNCTUATION & SPELLING: Punctuation, capitalization, or minor character tweaks.
  4. EXTRACTION ARTIFACTS: Page numbers, running headers, table keys, missing ligatures, line breaks.

Usage:
    python3 compare_extracted_texts.py file_a.txt file_b.txt [--mode {major,all}] [--min-words N] [--show-artifacts]
"""

import sys
import re
import argparse
import difflib

LIGATURES = ["ffi", "ffl", "ff", "fi", "fl"]
MINOR_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "for", "and", "or", "to",
    "by", "from", "with", "as", "is", "it", "this", "that"
}


# --------------------------------------------------------------------------
# Cleaning pipeline & Artifact Filters
# --------------------------------------------------------------------------

def read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _looks_like_header(stripped_line):
    """Heuristic for page numbers, running headers, chapter or book titles."""
    if len(stripped_line) > 70:
        return False
    if re.search(r"[.,;:!?\"']$", stripped_line):
        return False
    words = stripped_line.split()
    if not words:
        return False
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return False

    if stripped_line.isupper() and len(stripped_line) > 3:
        return True

    good = 0
    has_upper = False
    for w in alpha_words:
        first = next(c for c in w if c.isalpha())
        if first.isupper():
            good += 1
            has_upper = True
        elif w.lower() in MINOR_WORDS:
            good += 1
    return good == len(alpha_words) and has_upper and len(alpha_words) >= 2


def _looks_like_caption_or_credit(stripped_line):
    """Detect photo credits and figure caption lines."""
    if re.search(r"\(photo credit\s+[\w\d.]+\)", stripped_line, re.IGNORECASE):
        return True
    if re.fullmatch(r"Figure\s+\d+.*", stripped_line, re.IGNORECASE):
        return True
    return False


def _looks_like_table_artifact(line):
    """Detect page margin / sidebar artifacts like cipher alphabet keys and frequency tables."""
    lower = line.lower()
    if "plain alphabet" in lower or "cipher alphabet" in lower:
        return True
    if re.search(r"alphabet\s+[a-z\s–-]{10,}", lower):
        return True
    if lower == "letter percentage" or lower.startswith("letter percentage"):
        return True
    if re.fullmatch(r"[a-z]\s+\d+(\.\d+)?", line.strip(), re.IGNORECASE):
        return True
    if re.match(r"^table\s+\d+\s+this table of relative frequencies", lower):
        return True
    return False


def strip_headers_and_artifacts(text):
    """Remove standalone page-number lines, header/title lines, captions, and table key artifacts."""
    # Strip inline photo credit tags
    text = re.sub(r"\(photo credit\s+[\w\d.]+\)", "", text, flags=re.IGNORECASE)
    lines = text.split("\n")
    cleaned = []
    removed = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            cleaned.append(line)
            continue
        if re.fullmatch(r"\d{1,4}", stripped):
            removed.append(("page-number", stripped))
            continue
        if _looks_like_header(stripped):
            removed.append(("header/title", stripped))
            continue
        if _looks_like_caption_or_credit(stripped):
            removed.append(("caption/credit", stripped))
            continue
        if _looks_like_table_artifact(stripped):
            removed.append(("table/key-artifact", stripped))
            continue
        cleaned.append(line)
    return "\n".join(cleaned), removed


def fix_hyphenation(text):
    """Join words split across a line-break hyphen."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def normalize_typography(text):
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_pipeline(raw_text):
    text, removed_lines = strip_headers_and_artifacts(raw_text)
    text = fix_hyphenation(text)
    text = normalize_typography(text)
    tokens = re.findall(r"\S+", text)
    return tokens, removed_lines


# --------------------------------------------------------------------------
# Classification logic
# --------------------------------------------------------------------------

def _ligature_normalize(s):
    for lig in LIGATURES:
        s = s.replace(lig, "")
    return s


def _alnum_only(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def classify_hunk(a_words, b_words, min_major_words=10):
    """Classify diff hunk into:
       'artifact-spacing', 'artifact-ligature', 'punctuation',
       'vocab-edit', or 'major-structural'
    """
    a_str = "".join(a_words).lower()
    b_str = "".join(b_words).lower()

    if a_str == b_str:
        return "artifact-spacing"
    if _ligature_normalize(a_str) == _ligature_normalize(b_str):
        return "artifact-ligature"
    if _alnum_only(a_str) == _alnum_only(b_str):
        return "punctuation"

    max_words = max(len(a_words), len(b_words))
    if max_words >= min_major_words:
        return "major-structural"
    else:
        return "vocab-edit"


def is_trivial_match(words):
    """Checks if an 'equal' word sequence is trivial enough that surrounding
    edits should be coalesced into a single contiguous replacement.
    """
    if not words:
        return True
    if len(words) == 1:
        w = re.sub(r"[^a-zA-Z0-9]", "", words[0].lower())
        return w in MINOR_WORDS or len(w) <= 2 or not any(c.isalnum() for c in words[0])
    if len(words) <= 2:
        clean = [re.sub(r"[^a-zA-Z0-9]", "", w.lower()) for w in words]
        return all(w in MINOR_WORDS for w in clean)
    return False


def coalesce_opcodes(opcodes, tokens_a, tokens_b):
    """Merges adjacent non-equal opcodes when separated only by trivial stop-word matches.
    Prevents single stop words ('of', 'the') from fracturing sentence rewrites into
    false vocabulary edits.
    """
    merged = []
    for op in opcodes:
        tag, i1, i2, j1, j2 = op
        if not merged:
            merged.append(list(op))
            continue

        if tag != "equal" and len(merged) >= 2 and merged[-1][0] == "equal" and merged[-2][0] != "equal":
            eq_op = merged[-1]
            eq_words = tokens_a[eq_op[1]:eq_op[2]]
            if is_trivial_match(eq_words):
                # Pop the trivial equal block
                merged.pop()
                # Extend the preceding diff to encompass this diff and the trivial bridge
                merged[-1][2] = i2
                merged[-1][4] = j2
                # Update tag based on merged spans
                has_a = merged[-1][2] > merged[-1][1]
                has_b = merged[-1][4] > merged[-1][3]
                if has_a and has_b:
                    merged[-1][0] = "replace"
                elif has_a:
                    merged[-1][0] = "delete"
                else:
                    merged[-1][0] = "insert"
                continue

        merged.append(list(op))

    return [tuple(x) for x in merged]


def find_self_duplicate(tokens, span, min_len=12, threshold=0.85):
    start, end = span
    length = end - start
    if length < min_len:
        return None
    target = tokens[start:end]
    search_region = tokens[:start]
    if not search_region:
        return None
    sm = difflib.SequenceMatcher(None, search_region, target, autojunk=False)
    match = sm.find_longest_match(0, len(search_region), 0, len(target))
    if match.size >= length * threshold:
        return match
    return None


def compute_inner_diff(a_words, b_words):
    """Run an inner word-level diff between two versions of the same paragraph."""
    sm = difflib.SequenceMatcher(
        None, [w.lower() for w in a_words], [w.lower() for w in b_words], autojunk=False
    )
    inner_edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            inner_edits.append({
                "tag": tag,
                "a_words": a_words[i1:i2],
                "b_words": b_words[j1:j2],
                "a_span": (i1, i2),
                "b_span": (j1, j2)
            })
    return inner_edits


def reconcile_transpositions(major_diffs, min_similarity=0.45, min_words=15):
    """
    Pairs deleted hunks in A with inserted hunks in B that represent
    the same text block displaced or transposed around tables/floats.
    """
    deletes = [d for d in major_diffs if d["tag"] == "delete" and len(d["a_words"]) >= min_words]
    inserts = [i for i in major_diffs if i["tag"] == "insert" and len(i["b_words"]) >= min_words]

    matched_del_indices = set()
    matched_ins_indices = set()
    reordered_entries = []

    for d_idx, d in enumerate(deletes):
        best_sim = 0.0
        best_i_idx = None
        d_lower = [w.lower() for w in d["a_words"]]

        for i_idx, i in enumerate(inserts):
            if i_idx in matched_ins_indices:
                continue
            i_lower = [w.lower() for w in i["b_words"]]
            sim = difflib.SequenceMatcher(None, d_lower, i_lower, autojunk=False).ratio()
            if sim > best_sim:
                best_sim = sim
                best_i_idx = i_idx

        if best_sim >= min_similarity and best_i_idx is not None:
            matched_del_indices.add(d_idx)
            matched_ins_indices.add(best_i_idx)
            ins_entry = inserts[best_i_idx]
            inner_edits = compute_inner_diff(d["a_words"], ins_entry["b_words"])

            reordered_entries.append({
                "tag": "reordered",
                "kind": "major-structural",
                "similarity": best_sim,
                "a_span": d["a_span"],
                "b_span": ins_entry["b_span"],
                "a_words": d["a_words"],
                "b_words": ins_entry["b_words"],
                "inner_edits": inner_edits,
            })

    matched_del_objs = {id(deletes[idx]) for idx in matched_del_indices}
    matched_ins_objs = {id(inserts[idx]) for idx in matched_ins_indices}

    final_major = [d for d in major_diffs if id(d) not in matched_del_objs and id(d) not in matched_ins_objs]
    final_major.extend(reordered_entries)
    return final_major


# --------------------------------------------------------------------------
# Reporting & Output Formatting
# --------------------------------------------------------------------------

def context_str(tokens, start, end, n=6):
    before = " ".join(tokens[max(0, start - n):start])
    after = " ".join(tokens[end:end + n])
    return before, after


def print_hunk_details(idx, entry, path_a, path_b, tokens_a, tokens_b, context_n=6):
    tag = entry["tag"]
    kind = entry["kind"]
    a_words = entry["a_words"]
    b_words = entry["b_words"]
    a_before, a_after = context_str(tokens_a, *entry["a_span"], n=context_n)
    b_before, b_after = context_str(tokens_b, *entry["b_span"], n=context_n)

    if tag == "reordered":
        sim_pct = f"{int(round(entry.get('similarity', 0.0) * 100))}%"
        print(f"--- Difference #{idx} [REORDERED / REVISED PARAGRAPH] ({sim_pct} match, {len(a_words)} words in A vs {len(b_words)} words in B) ---")
        print(f"  * Note: This paragraph appears in different relative locations in File A and File B (e.g. transposed across a table/float).")
        print(f"  FILE A ({path_a}):")
        print(f"    ...{a_before}\n    >>> {' '.join(a_words)} <<<\n    {a_after}...")
        print(f"  FILE B ({path_b}):")
        print(f"    ...{b_before}\n    >>> {' '.join(b_words)} <<<\n    {b_after}...")
        inner = entry.get("inner_edits", [])
        if inner:
            print(f"  Inner revisions within paragraph ({len(inner)} edits):")
            for e in inner:
                itag = e["tag"]
                ia = " ".join(e["a_words"])
                ib = " ".join(e["b_words"])
                if itag == "replace":
                    print(f"    • [WORDING] \"{ia}\" <--> \"{ib}\"")
                elif itag == "delete":
                    print(f"    • [OMITTED IN B] \"{ia}\"")
                elif itag == "insert":
                    print(f"    • [ADDED IN B] \"{ib}\"")
        print()
        return

    word_count_str = ""
    if tag == "delete":
        word_count_str = f"[-{len(a_words)} words in B]"
    elif tag == "insert":
        word_count_str = f"[+{len(b_words)} words in B]"
    elif tag == "replace":
        word_count_str = f"[{len(a_words)} words in A vs {len(b_words)} words in B]"

    print(f"--- Difference #{idx} [{kind.upper()}] ({tag.upper()}) {word_count_str} ---")

    if tag in ("replace", "delete") and a_words:
        print(f"  FILE A ({path_a}):")
        if len(a_words) > 30:
            print(f"    ...{a_before}\n    >>> {' '.join(a_words)} <<<\n    {a_after}...")
        else:
            print(f"    ...{a_before} >>> {' '.join(a_words)} <<< {a_after}...")

    if tag in ("replace", "insert") and b_words:
        print(f"  FILE B ({path_b}):")
        if len(b_words) > 30:
            print(f"    ...{b_before}\n    >>> {' '.join(b_words)} <<<\n    {b_after}...")
        else:
            print(f"    ...{b_before} >>> {' '.join(b_words)} <<< {b_after}...")

    if tag == "insert":
        dup = find_self_duplicate(tokens_b, entry["b_span"])
        if dup:
            print(f"    [Note: Looks like a duplicate text box / pull-quote of ~{dup.size} words seen earlier in {path_b}]")
    elif tag == "delete":
        dup = find_self_duplicate(tokens_a, entry["a_span"])
        if dup:
            print(f"    [Note: Looks like a duplicate text box / pull-quote of ~{dup.size} words seen earlier in {path_a}]")
    print()
# --------------------------------------------------------------------------
# HTML Side-by-Side Exporter
# --------------------------------------------------------------------------

def generate_html_report(path_a, path_b, tokens_a, tokens_b, removed_a, removed_b, major_diffs, vocab_diffs, punct_diffs, artifact_diffs, html_path, min_major_words=10, context_n=6, max_diff_words=60):
    import html

    def render_words(words):
        return html.escape(" ".join(words))

    def render_diff_content(words, mark_cls, before, after):
        before_str = f'<span class="context">{html.escape(before)}</span> ' if before else ""
        after_str = f' <span class="context">{html.escape(after)}</span>' if after else ""

        if max_diff_words <= 0 or len(words) <= max_diff_words:
            content = f'<mark class="{mark_cls}">{html.escape(" ".join(words))}</mark>'
            return f"{before_str}{content}{after_str}".strip()

        head_n = min(30, len(words) // 2)
        tail_n = min(20, len(words) - head_n - 1)
        omitted = len(words) - head_n - tail_n
        if omitted <= 0:
            content = f'<mark class="{mark_cls}">{html.escape(" ".join(words))}</mark>'
            return f"{before_str}{content}{after_str}".strip()

        head_text = html.escape(" ".join(words[:head_n]))
        tail_text = html.escape(" ".join(words[-tail_n:]))
        omitted_badge = f'<span class="omitted-badge">… [{omitted:,} words omitted] …</span>'
        shortened_content = f'<mark class="{mark_cls}">{head_text} {omitted_badge} {tail_text}</mark>'

        full_text = html.escape(" ".join(words))
        details = (
            f'<details class="diff-details">'
            f'<summary>▶ Show full text ({len(words):,} words)</summary>'
            f'<div class="diff-details-body"><mark class="{mark_cls}">{full_text}</mark></div>'
            f'</details>'
        )
        return f"{before_str}{shortened_content}{after_str}\n{details}".strip()

    def render_side_by_side_rows(diff_list, section_id):
        rows_html = []
        for idx, entry in enumerate(diff_list, 1):
            tag = entry["tag"]
            kind = entry["kind"]
            a_words = entry["a_words"]
            b_words = entry["b_words"]
            a_before, a_after = context_str(tokens_a, *entry["a_span"], n=context_n)
            b_before, b_after = context_str(tokens_b, *entry["b_span"], n=context_n)

            if tag == "reordered":
                sim_pct = int(round(entry.get("similarity", 0.0) * 100))
                tag_label = "REORDERED"
                badge_cls = "badge-reorder"
                badge_text = f"Reordered ({sim_pct}% match)"

                sm_inner = difflib.SequenceMatcher(
                    None, [w.lower() for w in a_words], [w.lower() for w in b_words], autojunk=False
                )
                a_marked = []
                for itag, i1, i2, j1, j2 in sm_inner.get_opcodes():
                    chunk = render_words(a_words[i1:i2])
                    if itag == "equal":
                        a_marked.append(chunk)
                    else:
                        a_marked.append(f'<mark class="del-highlight">{chunk}</mark>')

                b_marked = []
                for itag, i1, i2, j1, j2 in sm_inner.get_opcodes():
                    chunk = render_words(b_words[j1:j2])
                    if itag == "equal":
                        b_marked.append(chunk)
                    else:
                        b_marked.append(f'<mark class="add-highlight">{chunk}</mark>')

                a_cell_content = f'<span class="context">{html.escape(a_before)}</span> ... {" ".join(a_marked)} ... <span class="context">{html.escape(a_after)}</span>'
                b_cell_content = f'<span class="context">{html.escape(b_before)}</span> ... {" ".join(b_marked)} ... <span class="context">{html.escape(b_after)}</span>'
                dup_note = f'<div class="dup-note">🔄 Reordered / Transposed Paragraph: Appears in different relative locations in File A and File B ({sim_pct}% text match). Inner revisions highlighted below.</div>'
            else:
                tag_label = tag.upper()
                badge_cls = "badge-del" if tag == "delete" else ("badge-add" if tag == "insert" else "badge-mod")
                if tag == "delete":
                    badge_text = f"-{len(a_words)} words"
                elif tag == "insert":
                    badge_text = f"+{len(b_words)} words"
                else:
                    badge_text = f"{len(a_words)} vs {len(b_words)} words"

                a_cell_content = ""
                if tag in ("replace", "delete") and a_words:
                    a_cell_content = render_diff_content(a_words, "del-highlight", a_before, a_after)
                else:
                    a_cell_content = f'<span class="context">{html.escape(a_before)} ... {html.escape(a_after)}</span> <span class="no-change">(No text in File A)</span>'

                b_cell_content = ""
                if tag in ("replace", "insert") and b_words:
                    b_cell_content = render_diff_content(b_words, "add-highlight", b_before, b_after)
                else:
                    b_cell_content = f'<span class="context">{html.escape(b_before)} ... {html.escape(b_after)}</span> <span class="no-change">(No text in File B)</span>'

                dup_note = ""
                if tag == "insert":
                    dup = find_self_duplicate(tokens_b, entry["b_span"])
                    if dup:
                        dup_note = f'<div class="dup-note">ℹ️ Duplicate section notice: matches ~{dup.size} words appearing earlier in File B</div>'
                elif tag == "delete":
                    dup = find_self_duplicate(tokens_a, entry["a_span"])
                    if dup:
                        dup_note = f'<div class="dup-note">ℹ️ Duplicate section notice: matches ~{dup.size} words appearing earlier in File A</div>'

            row = f'''
            <div class="diff-card {section_id}-card">
                <div class="card-header">
                    <span class="diff-num">#{idx}</span>
                    <span class="tag-badge {badge_cls}">{tag_label}</span>
                    <span class="word-badge">{badge_text}</span>
                </div>
                {dup_note}
                <div class="side-by-side">
                    <div class="pane pane-a">
                        <div class="pane-title">File A: {html.escape(path_a)}</div>
                        <div class="pane-body">{a_cell_content}</div>
                    </div>
                    <div class="pane pane-b">
                        <div class="pane-title">File B: {html.escape(path_b)}</div>
                        <div class="pane-body">{b_cell_content}</div>
                    </div>
                </div>
            </div>
            '''
            rows_html.append(row)
        return "\n".join(rows_html)

    major_rows = render_side_by_side_rows(major_diffs, "major")
    vocab_rows = render_side_by_side_rows(vocab_diffs, "vocab")

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Comparison: {html.escape(path_a)} vs {html.escape(path_b)}</title>
    <style>
        :root {{
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #212529;
            --text-muted: #6c757d;
            --border-color: #e9ecef;
            --border-dark: #dee2e6;
            --primary: #0d6efd;
            --del-bg: #ffebe9;
            --del-color: #cf222e;
            --add-bg: #e6ffec;
            --add-color: #1a7f37;
            --mod-bg: #fff8c5;
            --mod-color: #9a6700;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-dark);
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}

        h1 {{
            margin: 0 0 12px 0;
            font-size: 1.5rem;
            color: #111;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}

        .stat-box {{
            background: var(--bg-color);
            padding: 14px 18px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .stat-box .val {{
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .stat-box .lbl {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .filter-tabs {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }}

        .btn-tab {{
            background: var(--card-bg);
            border: 1px solid var(--border-dark);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .btn-tab.active {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}

        .section-header {{
            font-size: 1.25rem;
            font-weight: 700;
            margin: 32px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-dark);
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .diff-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-dark);
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }}

        .card-header {{
            background: #f1f3f5;
            padding: 10px 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .diff-num {{
            font-weight: 700;
            color: var(--text-muted);
        }}

        .tag-badge {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .badge-del {{ background: var(--del-bg); color: var(--del-color); }}
        .badge-add {{ background: var(--add-bg); color: var(--add-color); }}
        .badge-mod {{ background: var(--mod-bg); color: var(--mod-color); }}
        .badge-reorder {{ background: #d0ebff; color: #1971c2; }}

        .word-badge {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-left: auto;
            font-weight: 600;
        }}

        .dup-note {{
            background: #e7f5ff;
            color: #1971c2;
            padding: 8px 16px;
            font-size: 0.85rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .side-by-side {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            divide-x: 1px solid var(--border-color);
        }}

        @media (max-width: 900px) {{
            .side-by-side {{
                grid-template-columns: 1fr;
            }}
        }}

        .pane {{
            padding: 16px;
        }}

        .pane-a {{
            border-right: 1px solid var(--border-color);
        }}

        .pane-title {{
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .pane-body {{
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .context {{
            color: #6a737d;
        }}

        .del-highlight {{
            background-color: var(--del-bg);
            color: var(--del-color);
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 600;
            text-decoration: line-through;
        }}

        .add-highlight {{
            background-color: var(--add-bg);
            color: var(--add-color);
            padding: 2px 4px;
            border-radius: 4px;
            font-weight: 600;
        }}

        .no-change {{
            color: var(--text-muted);
            font-style: italic;
        }}

        .omitted-badge {{
            display: inline-block;
            background: rgba(0, 0, 0, 0.07);
            color: var(--text-muted);
            padding: 1px 7px;
            margin: 0 4px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            font-style: italic;
            border: 1px dashed rgba(0, 0, 0, 0.25);
            text-decoration: none !important;
        }}

        .diff-details {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed var(--border-dark);
            font-size: 0.85rem;
        }}

        .diff-details summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--primary);
            user-select: none;
            padding: 3px 0;
            outline: none;
        }}

        .diff-details summary:hover {{
            text-decoration: underline;
        }}

        .diff-details-body {{
            margin-top: 8px;
            padding: 12px;
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            max-height: 350px;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📄 Text Comparison Report</h1>
            <div>Comparing <strong>{html.escape(path_a)}</strong> vs <strong>{html.escape(path_b)}</strong></div>

            <div class="meta-grid">
                <div class="stat-box">
                    <div class="val">{len(major_diffs)}</div>
                    <div class="lbl">Major Edition Blocks</div>
                </div>
                <div class="stat-box">
                    <div class="val">{len(vocab_diffs)}</div>
                    <div class="lbl">Vocabulary Edits</div>
                </div>
                <div class="stat-box">
                    <div class="val">{len(artifact_diffs) + len(punct_diffs)}</div>
                    <div class="lbl">Artifacts Filtered</div>
                </div>
                <div class="stat-box">
                    <div class="val">{len(tokens_a)} / {len(tokens_b)}</div>
                    <div class="lbl">Words (File A / File B)</div>
                </div>
            </div>
        </header>

        <div class="filter-tabs">
            <button class="btn-tab active" onclick="showTab('all')">Show All</button>
            <button class="btn-tab" onclick="showTab('major')">Major Blocks ({len(major_diffs)})</button>
            <button class="btn-tab" onclick="showTab('vocab')">Vocab Edits ({len(vocab_diffs)})</button>
        </div>

        <div id="sec-major">
            <div class="section-header">
                📌 Major Edition-Level Content Blocks (Sorted by Size)
            </div>
            {major_rows if major_rows else '<p class="no-change">No major content additions or deletions found.</p>'}
        </div>

        <div id="sec-vocab">
            <div class="section-header">
                ✏️ Vocabulary & Word-Choice Edits
            </div>
            {vocab_rows if vocab_rows else '<p class="no-change">No minor vocabulary edits found.</p>'}
        </div>
    </div>

    <script>
        function showTab(mode) {{
            document.querySelectorAll('.btn-tab').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            const secMajor = document.getElementById('sec-major');
            const secVocab = document.getElementById('sec-vocab');

            if (mode === 'major') {{
                secMajor.style.display = 'block';
                secVocab.style.display = 'none';
            }} else if (mode === 'vocab') {{
                secMajor.style.display = 'none';
                secVocab.style.display = 'block';
            }} else {{
                secMajor.style.display = 'block';
                secVocab.style.display = 'block';
            }}
        }}
    </script>
</body>
</html>
'''

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"➜ Side-by-side HTML comparison report generated: {html_path}")


def compare(path_a, path_b, mode="all", min_major_words=10, show_artifacts=False, context_n=6, html_out=None, no_coalesce=False, max_diff_words=60):
    raw_a = read_file(path_a)
    raw_b = read_file(path_b)

    tokens_a, removed_a = clean_pipeline(raw_a)
    tokens_b, removed_b = clean_pipeline(raw_b)

    print("=" * 80)
    print(f" TEXT COMPARISON REPORT")
    print(f" File A: {path_a} ({len(tokens_a)} words, {len(removed_a)} layout/header lines removed)")
    print(f" File B: {path_b} ({len(tokens_b)} words, {len(removed_b)} layout/header lines removed)")
    print("=" * 80)
    print()

    sm = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    opcodes = sm.get_opcodes()
    if not no_coalesce:
        opcodes = coalesce_opcodes(opcodes, tokens_a, tokens_b)

    major_diffs = []
    vocab_diffs = []
    punct_diffs = []
    artifact_diffs = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        a_words = tokens_a[i1:i2]
        b_words = tokens_b[j1:j2]

        kind = classify_hunk(a_words, b_words, min_major_words=min_major_words)

        entry = {
            "tag": tag,
            "a_span": (i1, i2),
            "b_span": (j1, j2),
            "a_words": a_words,
            "b_words": b_words,
            "kind": kind,
        }

        if kind == "major-structural":
            major_diffs.append(entry)
        elif kind == "vocab-edit":
            vocab_diffs.append(entry)
        elif kind == "punctuation":
            punct_diffs.append(entry)
        else:
            artifact_diffs.append(entry)

    # Reconcile any displaced/transposed blocks between additions and deletions
    major_diffs = reconcile_transpositions(major_diffs, min_similarity=0.45, min_words=15)

    # Sort major diffs by size (max word count in either file) descending
    major_diffs.sort(key=lambda e: max(len(e["a_words"]), len(e["b_words"])), reverse=True)
    vocab_diffs.sort(key=lambda e: max(len(e["a_words"]), len(e["b_words"])), reverse=True)

    print("SUMMARY OF DIFFERENCES FOUND:")
    print(f"  • Major Structural Changes (>= {min_major_words} words): {len(major_diffs)}")
    print(f"  • Vocabulary / Minor Word Edits (< {min_major_words} words): {len(vocab_diffs)}")
    print(f"  • Punctuation / Typo Differences: {len(punct_diffs)}")
    print(f"  • Extraction Artifacts Filtered (ligatures/spacing/tables): {len(artifact_diffs)}")
    print()

    # 1. Major Structural Changes (Sorted Largest to Smallest)
    print("=" * 80)
    print(f" [SECTION 1] MAJOR EDITION-LEVEL CONTENT BLOCKS ({len(major_diffs)} found, sorted by size)")
    print("=" * 80 + "\n")

    if not major_diffs:
        print("  No major structural paragraph/section additions or deletions found.\n")
    else:
        for idx, entry in enumerate(major_diffs, 1):
            print_hunk_details(idx, entry, path_a, path_b, tokens_a, tokens_b, context_n=context_n)

    # 2. Vocabulary / Minor Word Edits
    if mode in ("all", "vocab"):
        print("=" * 80)
        print(f" [SECTION 2] VOCABULARY & WORD-CHOICE EDITS ({len(vocab_diffs)} found)")
        print("=" * 80 + "\n")

        if not vocab_diffs:
            print("  No minor word-level edits found.\n")
        else:
            for idx, entry in enumerate(vocab_diffs, 1):
                print_hunk_details(idx, entry, path_a, path_b, tokens_a, tokens_b, context_n=context_n)
    else:
        print(f"  (Skipping {len(vocab_diffs)} minor vocabulary edits in '--mode major' mode. Use '--mode all' to display them)\n")

    # 3. Artifacts
    if show_artifacts:
        print("=" * 80)
        print(f" [SECTION 3] FILTERED ARTIFACTS ({len(artifact_diffs) + len(punct_diffs)} total)")
        print("=" * 80 + "\n")
        for entry in punct_diffs + artifact_diffs:
            print(f"  [{entry['kind']}] A: {' '.join(entry['a_words'])} <-> B: {' '.join(entry['b_words'])}")
        print()

    if html_out:
        generate_html_report(
            path_a, path_b, tokens_a, tokens_b, removed_a, removed_b,
            major_diffs, vocab_diffs, punct_diffs, artifact_diffs,
            html_out, min_major_words=min_major_words, context_n=context_n,
            max_diff_words=max_diff_words
        )


def main():
    parser = argparse.ArgumentParser(description="Diff PDF extracted text files, prioritizing major structural changes over word edits.")
    parser.add_argument("file_a", help="Path to first text file")
    parser.add_argument("file_b", help="Path to second text file")
    parser.add_argument("--mode", choices=["major", "vocab", "all"], default="all", help="Display mode: 'major' shows only big structural changes; 'all' shows both major changes and vocab edits")
    parser.add_argument("--min-words", type=int, default=10, help="Minimum number of words in a hunk to qualify as a major structural change (default: 10)")
    parser.add_argument("--show-artifacts", action="store_true", help="List details of filtered ligatures/spacing/table artifacts")
    parser.add_argument("--context", type=int, default=6, help="Number of context words to show around each diff")
    parser.add_argument("--html", help="Path to output side-by-side HTML comparison report file (e.g. diff_report.html)")
    parser.add_argument(
        "--no-coalesce",
        action="store_true",
        help="Disable automatic merging of diff hunks separated by trivial stop words (e.g. 'of', 'the')",
    )
    parser.add_argument(
        "--max-diff-words",
        type=int,
        default=60,
        help="Maximum words to display for a diff/insert before shortening in HTML (default: 60, use 0 to disable)",
    )
    args = parser.parse_args()

    compare(
        args.file_a,
        args.file_b,
        mode=args.mode,
        min_major_words=args.min_words,
        show_artifacts=args.show_artifacts,
        context_n=args.context,
        html_out=args.html,
        no_coalesce=args.no_coalesce,
        max_diff_words=args.max_diff_words,
    )


if __name__ == "__main__":
    main()
