"""Mirror an upstream HF dataset into the private sage2 repo for a benchmark.

    HF_TOKEN=<write token> uv run scripts/mirror_dataset.py swebench-verified \
        SWE-bench/SWE-bench_Verified --org <org> --resource-group-id <id>

Creates ``<org>/sage2-<benchmark-id>`` (private), uploads the upstream snapshot
unchanged and records the upstream repo + commit in the commit message and a
SAGE2_SOURCE.json file, so every mirror revision traces back to its source.
Prints the mirror revision to pin in the granite.build step config.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id")
    p.add_argument("upstream", help="upstream dataset repo id")
    p.add_argument("--upstream-revision", default=None)
    p.add_argument("--org", required=True)
    p.add_argument("--resource-group-id", default=None, help="HF enterprise resource group for the repo")
    args = p.parse_args()

    api = HfApi()
    upstream_sha = api.dataset_info(args.upstream, revision=args.upstream_revision).sha
    repo_id = f"{args.org}/sage2-{args.benchmark_id}"
    api.create_repo(
        repo_id, repo_type="dataset", private=True, exist_ok=True, resource_group_id=args.resource_group_id
    )
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(
            snapshot_download(args.upstream, repo_type="dataset", revision=upstream_sha, local_dir=Path(tmp) / "ds")
        )
        (local / "SAGE2_SOURCE.json").write_text(
            json.dumps({"benchmark": args.benchmark_id, "upstream": args.upstream, "revision": upstream_sha}, indent=2)
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=local,
            ignore_patterns=[".cache/**"],
            commit_message=f"Mirror {args.upstream}@{upstream_sha}",
        )
    print(f"{repo_id}@{commit.oid}  (from {args.upstream}@{upstream_sha})")


if __name__ == "__main__":
    main()
