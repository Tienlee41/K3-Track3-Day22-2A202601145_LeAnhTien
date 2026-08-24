from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from gpu_common import load_gpu_config, require_gpu, save_json, seed_everything
from model_inference import clear_cuda, generate, load_variant

TASKS = [
    {
        "suite": "IFEval-lite",
        "prompt": (
            "Trả lời đúng 3 gạch đầu dòng, mỗi dòng tối đa 8 từ: lợi ích của kiểm thử phần mềm."
        ),
        "expected": "three_short_bullets",
    },
    {
        "suite": "IFEval-lite",
        "prompt": "Trả lời chỉ bằng một câu có đúng 12 từ tiếng Việt về năng lượng mặt trời.",
        "expected": "twelve_words",
    },
    {
        "suite": "GSM8K-lite",
        "prompt": (
            "Lan có 15 quả cam, cho bạn 1/3 rồi mua thêm 7 quả. "
            "Lan còn bao nhiêu quả? Chỉ ghi số cuối cùng."
        ),
        "expected": "17",
    },
    {
        "suite": "GSM8K-lite",
        "prompt": (
            "Một xe đi 60 km/h trong 2.5 giờ. Quãng đường là bao nhiêu km? Chỉ ghi số cuối cùng."
        ),
        "expected": "150",
    },
    {
        "suite": "MMLU-lite",
        "prompt": (
            "Đâu là đơn vị SI của lực? A. Joule B. Newton C. Watt D. Pascal. Chỉ trả lời chữ cái."
        ),
        "expected": "B",
    },
    {
        "suite": "MMLU-lite",
        "prompt": (
            "Overfitting thường biểu hiện thế nào? A. Train thấp, test cao "
            "B. Cả hai cao C. Train cao, test thấp D. Không học. "
            "Chỉ trả lời chữ cái."
        ),
        "expected": "C",
    },
    {
        "suite": "AlpacaEval-lite",
        "prompt": "Giải thích ngắn gọn cách lập kế hoạch học Python trong 4 tuần cho người mới.",
        "expected": "helpful",
    },
    {
        "suite": "AlpacaEval-lite",
        "prompt": "Nêu ba cách kiểm chứng một thông tin đáng ngờ trên mạng.",
        "expected": "helpful",
    },
]


def _score(task: dict[str, str], response: str) -> float:
    expected = task["expected"]
    stripped = response.strip()
    if expected == "three_short_bullets":
        lines = [line for line in stripped.splitlines() if line.strip()]
        return float(len(lines) == 3 and all(len(line.split()) <= 9 for line in lines))
    if expected == "twelve_words":
        return float(len(re.findall(r"\b\w+\b", stripped, flags=re.UNICODE)) == 12)
    if expected == "helpful":
        words = len(stripped.split())
        structure = stripped.count("\n") + stripped.count("-") + stripped.count("1.")
        return min(1.0, words / 80 + min(structure, 3) * 0.15)
    if expected in {"B", "C"}:
        match = re.search(r"\b([ABCD])\b", stripped.upper())
        return float(bool(match and match.group(1) == expected))
    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", stripped)
    return float(bool(numbers and numbers[-1].replace(",", ".") == expected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four lightweight alignment benchmarks")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    output = Path(config["paths"]["benchmark"])
    if args.reuse and output.exists():
        print(f"Reusing {output}")
        return
    require_gpu()
    seed_everything(int(config["seed"]))

    all_results = []
    for variant in ("sft", "dpo"):
        model, tokenizer = load_variant(config, variant)
        for task in TASKS:
            response = generate(model, tokenizer, task["prompt"], 120)
            all_results.append(
                {**task, "model": variant, "response": response, "score": _score(task, response)}
            )
        del model, tokenizer
        clear_cuda()

    suites = sorted({task["suite"] for task in TASKS})
    summary = {
        variant: {
            suite: round(
                sum(
                    row["score"]
                    for row in all_results
                    if row["model"] == variant and row["suite"] == suite
                )
                / sum(1 for task in TASKS if task["suite"] == suite),
                3,
            )
            for suite in suites
        }
        for variant in ("sft", "dpo")
    }
    payload = {
        "scope": "2-item smoke subset per suite; not official leaderboard scores",
        "results": all_results,
        "summary": summary,
    }
    save_json(output, payload)

    x = range(len(suites))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        [value - width / 2 for value in x], [summary["sft"][s] for s in suites], width, label="SFT"
    )
    axis.bar(
        [value + width / 2 for value in x], [summary["dpo"][s] for s in suites], width, label="DPO"
    )
    axis.set_xticks(list(x), suites, rotation=12)
    axis.set_ylim(0, 1.05)
    axis.set(title="Alignment benchmark smoke subsets", ylabel="score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    screenshot = Path(config["paths"]["screenshots"]) / "06_benchmark_summary.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(screenshot, dpi=180)
    plt.close(fig)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
