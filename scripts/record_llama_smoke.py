from __future__ import annotations

import json
import re
from pathlib import Path


def looks_like_vietnamese(response: str) -> bool:
    diacritics = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return len(response.split()) >= 8 and any(char.casefold() in diacritics for char in response)


def main() -> None:
    log_path = Path("submission/llama_smoke.stdout.txt")
    text = log_path.read_text(encoding="utf-8")
    match = re.search(
        r"<\|im_start\|>assistant\s*(.*?)\s*\[\s*Prompt:",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"could not find generated response in {log_path}")
    response = " ".join(match.group(1).split())
    gguf = Path("gguf/lab22-dpo-Q4_K_M.gguf")
    payload = {
        "llama_cpp_tag": "b10566",
        "gguf": str(gguf),
        "size_bytes": gguf.stat().st_size,
        "size_gb": round(gguf.stat().st_size / 1024**3, 3),
        "quantization": "Q4_K_M",
        "prompt": "Viết một câu tiếng Việt giải thích học máy là gì.",
        "response": response,
        "coherent_vietnamese": looks_like_vietnamese(response),
        "stdout_log": str(log_path),
    }
    if not payload["coherent_vietnamese"]:
        raise RuntimeError("llama.cpp response did not pass the Vietnamese coherence smoke check")
    target = Path("submission/deploy.json")
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
