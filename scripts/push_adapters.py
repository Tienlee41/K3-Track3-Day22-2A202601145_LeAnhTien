from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token


def load_config(path: Path = Path("configs/gpu.yaml")) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    token = get_token()
    if not token:
        raise SystemExit("Hugging Face authentication is required for the professional bonus")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    config = load_config()
    repositories = {
        "sft": f"{username}/day22-qwen25-3b-sft-mini",
        "dpo": f"{username}/day22-qwen25-3b-dpo",
    }
    urls = {}
    revisions = {}
    for name, repo_id in repositories.items():
        source = Path(config["paths"][f"{name}_adapter"])
        if not (source / "adapter_config.json").exists():
            raise FileNotFoundError(f"missing adapter: {source}")
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=source,
            commit_message=f"Upload Day 22 {name.upper()} LoRA adapter",
        )
        info = api.model_info(repo_id)
        if info.private:
            raise RuntimeError(f"professional bonus repository is not public: {repo_id}")
        urls[name] = f"https://huggingface.co/{repo_id}"
        revisions[name] = info.sha
    payload = {
        "owner": username,
        "repositories": urls,
        "revisions": revisions,
        "visibility": "public",
    }
    save_json("submission/huggingface.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
