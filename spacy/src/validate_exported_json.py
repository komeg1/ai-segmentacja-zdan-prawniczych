"""Walidacja exported_sentences_ls_merged.json — overlaps, gaps, długie zdania itd."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "../data/exported_sentences_ls/exported_sentences_ls_merged.json"

GAP_THRESHOLD = 5000
LONG_SENT_THRESHOLD = 1000


def is_list_item(text: str) -> bool:
    t = text.lstrip("\r\n")
    return bool(re.match(r"^[a-z]\)", t)) or bool(re.match(r"^\d+[a-z]*\)", t))


def validate_document(doc: dict) -> dict:
    act_id = doc["act_id"]
    sents = sorted(doc["sentences"], key=lambda s: s["start_offset"])
    issues = {
        "overlaps": [],
        "gaps": [],
        "long_sentences": [],
        "lowercase_end_no_punct": [],
        "lowercase_start_not_list": [],
        "invalid_range": [],
    }

    for i, s in enumerate(sents):
        start, end = s["start_offset"], s["end_offset"]
        text = s["text"]
        length = end - start

        if start >= end:
            issues["invalid_range"].append((i, start, end))

        for j, other in enumerate(sents):
            if i >= j:
                continue
            os, oe = other["start_offset"], other["end_offset"]
            if start < oe and end > os:
                issues["overlaps"].append({
                    "sent_a": i,
                    "sent_b": j,
                    "range_a": [start, end],
                    "range_b": [os, oe],
                    "relation": _overlap_relation(start, end, os, oe),
                })

        if length > LONG_SENT_THRESHOLD:
            issues["long_sentences"].append({
                "idx": i,
                "start": start,
                "end": end,
                "length": length,
                "preview": text[:100].replace("\r", " ").replace("\n", " "),
            })

        t = text.rstrip("\r\n")
        if t and t[-1].islower() and t[-1] not in ")]":
            issues["lowercase_end_no_punct"].append({
                "idx": i,
                "start": start,
                "preview": text[:100].replace("\r", " ").replace("\n", " "),
            })

        t2 = text.lstrip("\r\n")
        if t2 and t2[0].islower() and not is_list_item(t2):
            issues["lowercase_start_not_list"].append({
                "idx": i,
                "start": start,
                "preview": text[:100].replace("\r", " ").replace("\n", " "),
            })

    for i in range(len(sents) - 1):
        gap = sents[i + 1]["start_offset"] - sents[i]["end_offset"]
        if gap > GAP_THRESHOLD:
            issues["gaps"].append({
                "after_sent": i,
                "gap_chars": gap,
                "from_offset": sents[i]["end_offset"],
                "to_offset": sents[i + 1]["start_offset"],
            })

    return {"act_id": act_id, "sentence_count": len(sents), "issues": issues}


def _overlap_relation(a0, a1, b0, b1) -> str:
    if a0 <= b0 and a1 >= b1:
        return "A_contains_B"
    if b0 <= a0 and b1 >= a1:
        return "B_contains_A"
    return "partial_overlap"


def has_critical_issues(issues: dict) -> bool:
    return bool(
        issues["overlaps"]
        or issues["gaps"]
        or issues["invalid_range"]
    )


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    data = json.load(open(json_path, encoding="utf-8"))

    results = [validate_document(doc) for doc in data]

    categories = [
        "overlaps",
        "gaps",
        "long_sentences",
        "lowercase_end_no_punct",
        "lowercase_start_not_list",
        "invalid_range",
    ]

    by_cat = {c: [] for c in categories}
    critical_acts = []
    any_issue_acts = []

    for r in results:
        act = r["act_id"]
        iss = r["issues"]
        if any(iss[c] for c in categories):
            any_issue_acts.append(act)
        if has_critical_issues(iss):
            critical_acts.append(act)
        for c in categories:
            if iss[c]:
                by_cat[c].append(act)

    print(f"Plik: {json_path}")
    print(f"Dokumentow: {len(data)}")
    print(f"Aktow z jakimkolwiek problemem: {len(any_issue_acts)}")
    print(f"Aktow z problemami KRYTYCZNYMI (overlaps/gaps): {len(critical_acts)}")
    print()

    for c in categories:
        print(f"{c}: {len(by_cat[c])} aktow")

    print("\n=== AKTY Z PROBLEMAMI KRYTYCZNYMI ===")
    for act in sorted(critical_acts):
        r = next(x for x in results if x["act_id"] == act)
        iss = r["issues"]
        parts = [c for c in ["overlaps", "gaps", "invalid_range"] if iss[c]]
        print(f"\n{act} ({', '.join(parts)})")
        for ov in iss["overlaps"][:5]:
            print(
                f"  overlap {ov['relation']}: "
                f"#{ov['sent_a']} {ov['range_a']} vs #{ov['sent_b']} {ov['range_b']}"
            )
        if len(iss["overlaps"]) > 5:
            print(f"  ... +{len(iss['overlaps']) - 5} wiecej overlapow")
        for g in iss["gaps"]:
            print(f"  gap {g['gap_chars']} znakow: {g['from_offset']} -> {g['to_offset']}")

    print("\n=== WSZYSTKIE AKTY Z JAKIMKOLWIEK PROBLEMEM ===")
    for act in sorted(any_issue_acts):
        r = next(x for x in results if x["act_id"] == act)
        iss = r["issues"]
        flags = []
        if iss["overlaps"]:
            flags.append(f"overlaps={len(iss['overlaps'])}")
        if iss["gaps"]:
            flags.append(f"gaps={len(iss['gaps'])}")
        if iss["long_sentences"]:
            flags.append(f"long={len(iss['long_sentences'])}")
        if iss["lowercase_end_no_punct"]:
            flags.append(f"lc_end={len(iss['lowercase_end_no_punct'])}")
        if iss["lowercase_start_not_list"]:
            flags.append(f"lc_start={len(iss['lowercase_start_not_list'])}")
        print(f"  {act}: {', '.join(flags)}")

    out = json_path.parent / f"validation_report_{json_path.stem}.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nRaport JSON: {out}")


if __name__ == "__main__":
    main()
