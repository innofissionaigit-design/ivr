#!/usr/bin/env python3
"""Generate the stimulus set and answer key for the figure listening test.

story title: Figures are spoken at a pace a caller can write down
user story: As a patient noting a price or a reference, I want it grouped and
    slower, so that I do not have to ask twice.
acceptance criteria: Prices, phone numbers and reference identifiers are
    spoken with grouping and a reduced rate through the per-request speed
    parameter. A listening test confirms callers transcribe correctly on
    first hearing.

Emits TEXT, not audio. Rendering happens on the pod through the real TTS --
see docs/listening-test.md -- because a clip produced by anything else is not
the clip the caller hears, and the whole study is about what the caller hears.

Every stimulus is a figure inside an ordinary carrier sentence taken from the
real templates. A bare number tests something the service never does.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.bn_normalize import GROUPING_POLICY_VERSION, verbalize  # noqa: E402

# Carrier sentences, one per figure type, shaped like the real templates in
# agent/reply_templates.py. {} is where the figure goes.
CARRIERS = {
    "price": "ইউরিক অ্যাসিড টেস্টের রেট {} টাকা।",
    "phone": "আপনার দেওয়া ফোন নম্বর {}।",
    "reference": "আপনার কনফার্মেশন নম্বর {}।",
}

# Spread rather than round: 4-digit and 5-digit prices are where a caller is
# most likely to mishear a magnitude, and that is the expensive mistake.
PRICES = ["250", "650", "1250", "1800", "2200", "7", "12500"]

PHONES = ["9876543210", "9331045678", "8420076543", "6291100045"]

REFERENCES = [
    "KCD-20260914-4A2F",
    "KCD-20260914-0031",
    "KCD-20261231-FFFF",
    "KCD-20260101-B7C0",
]


def _stimuli() -> list[dict]:
    out: list[dict] = []
    for kind, values in (("price", PRICES), ("phone", PHONES),
                         ("reference", REFERENCES)):
        for value in values:
            sentence = CARRIERS[kind].format(value)
            out.append({
                "id": f"{kind}-{value}",
                "kind": kind,
                "expected": value,
                "text": sentence,
                # What the synthesiser will actually be handed. Recorded so a
                # result can be read back later without re-running the code
                # that produced it -- the grouping is visible right here.
                "spoken": verbalize(sentence),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="stimuli", help="output directory")
    parser.add_argument("--seed", type=int, default=20260912,
                        help="shuffle seed; recorded in the manifest so an "
                             "order can be reproduced")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = _stimuli()
    random.Random(args.seed).shuffle(stimuli)

    manifest = {
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "seed": args.seed,
        "count": len(stimuli),
        "stimuli": stimuli,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # The answer key is separate and CSV, because the person scoring is
    # working in a spreadsheet and should never have to open the manifest --
    # which contains the spoken form and would give the answer away.
    with (out_dir / "answer_key.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "kind", "expected"])
        for s in stimuli:
            writer.writerow([s["id"], s["kind"], s["expected"]])

    print(f"{len(stimuli)} stimuli -> {out_dir}/manifest.json")
    print(f"answer key       -> {out_dir}/answer_key.csv")
    print(f"grouping policy  -> {GROUPING_POLICY_VERSION}")
    print("render on the pod; see docs/listening-test.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
