from __future__ import annotations

# ruff: noqa: I001 -- compatibility import must precede third-party GPU imports

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import windows_platform_compat  # noqa: F401  # must precede torch/PEFT imports
import urllib.request
import zipfile

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_common import load_gpu_config, require_gpu, save_json


LLAMA_TAG = "b10566"
LLAMA_RAW_BASE = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_TAG}"
LLAMA_BIN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
    f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"
)
PYTHON_SCRIPT_RUNNER = (
    "import runpy,sys; from pathlib import Path; "
    "sys.path.insert(0, 'scripts'); "
    "import windows_platform_compat; "
    "script=sys.argv.pop(1); sys.argv[0]=script; "
    "sys.path.insert(0, str(Path(script).resolve().parent)); "
    "runpy.run_path(script, run_name='__main__')"
)


def _remove_readonly(function: Callable[[str], None], path: str, exc_info: object) -> None:
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, onerror=_remove_readonly)


def _download_range(url: str, start: int, end: int, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        if response.status != 206:
            raise OSError(f"range request returned HTTP {response.status}")
        shutil.copyfileobj(response, output)


def _download_raw(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        return
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {url}")
    part_paths: list[Path] = []
    try:
        probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(probe) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                raise OSError("server does not advertise byte ranges")
            total = int(content_range.rsplit("/", 1)[1])
            final_url = response.geturl()
        workers = 8
        chunk = (total + workers - 1) // workers
        ranges = [(start, min(start + chunk - 1, total - 1)) for start in range(0, total, chunk)]
        part_paths = [
            partial.with_suffix(partial.suffix + f".{index}") for index in range(len(ranges))
        ]
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_download_range, final_url, start, end, part_path)
                for (start, end), part_path in zip(ranges, part_paths)
            ]
            for future in futures:
                future.result()
        with partial.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as part:
                    shutil.copyfileobj(part, output)
        print(f"Downloaded {total / 1024**2:.1f} MiB using {len(ranges)} parallel ranges")
    except (OSError, ValueError) as exc:
        print(f"Parallel download unavailable ({exc}); falling back to a single stream")
        partial.unlink(missing_ok=True)
        urllib.request.urlretrieve(url, partial)
    finally:
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)
    partial.replace(destination)


def _prepare_llama_cpp() -> Path:
    root = Path("tools/llama.cpp")
    source_marker = root / ".source_qwen_gguf_complete"
    if source_marker.exists() and list(root.rglob("llama-quantize.exe")):
        return root

    downloads = Path("tools/downloads")
    binary_zip = downloads / f"llama-{LLAMA_TAG}-cpu.zip"
    if not source_marker.exists():
        if root.exists():
            _remove_tree(root)
        print(f"Downloading minimal llama.cpp {LLAMA_TAG} Qwen converter sources")
        source_files = [
            "convert_hf_to_gguf.py",
            "conversion/__init__.py",
            "conversion/base.py",
            "conversion/qwen.py",
            "gguf-py/gguf/__init__.py",
            "gguf-py/gguf/constants.py",
            "gguf-py/gguf/gguf.py",
            "gguf-py/gguf/gguf_reader.py",
            "gguf-py/gguf/gguf_writer.py",
            "gguf-py/gguf/lazy.py",
            "gguf-py/gguf/metadata.py",
            "gguf-py/gguf/py.typed",
            "gguf-py/gguf/quants.py",
            "gguf-py/gguf/tensor_mapping.py",
            "gguf-py/gguf/utility.py",
            "gguf-py/gguf/vocab.py",
        ]
        for relative in source_files:
            _download_raw(f"{LLAMA_RAW_BASE}/{relative}", root / relative)
        source_marker.write_text("minimal Qwen converter sources complete\n", encoding="utf-8")
    _download(LLAMA_BIN_URL, binary_zip)
    with zipfile.ZipFile(binary_zip) as archive:
        archive.extractall(root / "bin")
    return root


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", json.dumps(command, ensure_ascii=True))
    result = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def _looks_like_vietnamese(response: str) -> bool:
    diacritics = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    return len(response.split()) >= 8 and any(char.casefold() in diacritics for char in response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adapters, export Q4_K_M GGUF, smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--cpu-merge",
        action="store_true",
        help="merge the upstream BF16 checkpoint on CPU (export fallback)",
    )
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    if not args.cpu_merge:
        require_gpu()
    quantized = Path("gguf/lab22-dpo-Q4_K_M.gguf")
    metrics_path = Path("submission/deploy.json")
    if args.reuse and quantized.exists() and metrics_path.exists():
        print(f"Reusing {quantized}")
        return

    merged_dir = Path("gguf/merged-hf")
    merge_marker = merged_dir / ".merge_dequantized_complete"
    if merged_dir.exists() and not merge_marker.exists():
        print(f"Discarding incomplete merged model: {merged_dir}")
        shutil.rmtree(merged_dir)
    if not merge_marker.exists():
        export_base = config.get("export_base_model", config["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(export_base)
        base = AutoModelForCausalLM.from_pretrained(
            export_base,
            device_map=None if args.cpu_merge else {"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        sft = PeftModel.from_pretrained(base, config["paths"]["sft_adapter"]).merge_and_unload()
        if hasattr(sft, "peft_config"):
            delattr(sft, "peft_config")
        dpo = PeftModel.from_pretrained(sft, config["paths"]["dpo_adapter"]).merge_and_unload()
        if hasattr(dpo, "peft_config"):
            delattr(dpo, "peft_config")
        # Training uses the laptop-friendly 4-bit base. Export instead starts
        # from the compatible upstream BF16 checkpoint, so the converter sees
        # ordinary linear layers rather than bitsandbytes-specific modules.
        merged_dir.mkdir(parents=True, exist_ok=True)
        dpo.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="1GB")
        tokenizer.save_pretrained(merged_dir)
        merge_marker.write_text(
            "SFT merged, DPO merged, bitsandbytes layers dequantized\n", encoding="utf-8"
        )
        del dpo, sft, base
        torch.cuda.empty_cache()

    llama_root = _prepare_llama_cpp()
    converter = llama_root / "convert_hf_to_gguf.py"
    quantizer = next(llama_root.rglob("llama-quantize.exe"))
    cli = next(llama_root.rglob("llama-cli.exe"))
    f16 = Path("gguf/lab22-dpo-f16.gguf")
    if not f16.exists():
        f16_partial = f16.with_name(f"{f16.stem}.partial{f16.suffix}")
        f16_partial.unlink(missing_ok=True)
        result = _run(
            [
                sys.executable,
                "-c",
                PYTHON_SCRIPT_RUNNER,
                str(converter),
                str(merged_dir),
                "--outfile",
                str(f16_partial),
                "--outtype",
                "f16",
            ]
        )
        print(result.stdout)
        f16_partial.replace(f16)
    if not quantized.exists():
        quantized_partial = quantized.with_name(
            f"{quantized.stem}.partial{quantized.suffix}"
        )
        quantized_partial.unlink(missing_ok=True)
        result = _run([str(quantizer), str(f16), str(quantized_partial), "Q4_K_M"])
        print(result.stdout)
        quantized_partial.replace(quantized)

    prompt = "Viết một câu tiếng Việt giải thích học máy là gì."
    formatted_prompt = (
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    smoke = _run(
        [
            str(cli),
            "-m",
            str(quantized),
            "-p",
            formatted_prompt,
            "-n",
            "48",
            "-t",
            "8",
            "--temp",
            "0",
            "--no-display-prompt",
        ]
    )
    response = smoke.stdout.strip()
    payload = {
        "llama_cpp_tag": LLAMA_TAG,
        "gguf": str(quantized),
        "size_bytes": quantized.stat().st_size,
        "size_gb": round(quantized.stat().st_size / 1024**3, 3),
        "quantization": "Q4_K_M",
        "prompt": prompt,
        "response": response,
        "coherent_vietnamese": _looks_like_vietnamese(response),
    }
    save_json(metrics_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
