"""Compare FIFO and bounded priority on identical synthetic reading lists."""
import argparse
import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Study:
    arrival: int
    duration: int
    positive: bool
    flagged: bool


def generate(seed, arrival_probability, prevalence, sensitivity, specificity):
    rng = random.Random(seed)
    studies = []
    for minute in range(480):
        if rng.random() < arrival_probability:
            positive = rng.random() < prevalence
            flag_probability = sensitivity if positive else 1 - specificity
            studies.append(Study(minute, rng.randint(4, 8), positive,
                                 rng.random() < flag_probability))
    return studies


def evaluate(studies, prioritize, max_wait=60):
    upcoming = deque(studies)
    waiting = deque()
    outcomes = []
    minute = 0
    while upcoming or waiting:
        if not waiting:
            minute = max(minute, upcoming[0].arrival)
        while upcoming and upcoming[0].arrival <= minute:
            waiting.append(upcoming.popleft())
        index = 0
        if prioritize and minute - waiting[0].arrival < max_wait:
            index = next((i for i, study in enumerate(waiting) if study.flagged), 0)
        study = waiting[index]
        del waiting[index]
        outcomes.append((study, minute - study.arrival, minute + study.duration))
        minute += study.duration
    return outcomes


def summarize(outcomes):
    waits = [wait for _, wait, _ in outcomes]
    positives = [wait for study, wait, _ in outcomes if study.positive]
    missed = [wait for study, wait, _ in outcomes if study.positive and not study.flagged]
    return {
        "studies": len(outcomes),
        "synthetic_positives": len(positives),
        "false_negatives": len(missed),
        "mean_wait": mean(waits) if waits else None,
        "positive_mean_wait": mean(positives) if positives else None,
        "false_negative_mean_wait": mean(missed) if missed else None,
        "p90_wait": sorted(waits)[math.ceil(.9 * len(waits)) - 1] if waits else None,
        "reading_minutes": sum(study.duration for study, _, _ in outcomes),
        "completed_by_shift_end": sum(end <= 480 for _, _, end in outcomes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "moderate": (.14, .05, .90, .80),
        "backlog": (.25, .05, .90, .80),
        "lower_model_quality": (.25, .05, .65, .65),
    }
    rows = []
    summary = []
    for scenario, params in scenarios.items():
        scenario_rows = []
        for seed in range(100):
            studies = generate(seed, *params)
            fixed, priority = (evaluate(studies, p) for p in (False, True))
            assert len(fixed) == len(priority) == len(studies)
            assert sum(s.duration for s, _, _ in fixed) == sum(s.duration for s, _, _ in priority)
            for policy, outcomes in (("fifo", fixed), ("priority", priority)):
                scenario_rows.append(dict(scenario=scenario, seed=seed, policy=policy,
                                          **summarize(outcomes)))
        rows.extend(scenario_rows)
        for policy in ("fifo", "priority"):
            subset = [row for row in scenario_rows if row["policy"] == policy]
            averages = {}
            for field in summarize([]):
                values = [row[field] for row in subset if row[field] is not None]
                averages[field] = round(mean(values), 2) if values else None
            summary.append(dict(scenario=scenario, policy=policy, **averages))
    for name, table in (("runs", rows), ("summary", summary)):
        with (args.out / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=table[0])
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
