from __future__ import annotations

import json
from pathlib import Path

from gpu_common import load_gpu_config, save_json
from huggingface_hub import HfApi, get_token


def main() -> None:
    token = get_token()
    if not token:
        raise SystemExit("Hugging Face authentication is required for the professional bonus")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    config = load_gpu_config()
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
