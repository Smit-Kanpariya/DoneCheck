"""Download the public run files DoneCheck uses (they are not stored in this repository).

    python -m donecheck.fetch building data        # the five building models, about 800 MiB
    python -m donecheck.fetch final data_final     # GLM-4.5, the one-time final test

Per model it saves the leaderboard's per_instance_details.json and metadata.yaml (GitHub), its
exit_statuses_*.yaml files and every run file (public S3) as trajs/<bug>.traj.json. Files already on
disk with the listed size are kept, so it can be rerun after a failure. Every download is checked
against the size in the storage listing, and every run file is parsed once at the end.
"""

import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S3 = "https://swe-bench-submissions.s3.amazonaws.com"
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
ENTRIES = {
    "building": ["20250807_mini-v1.7.0_gpt-oss-120b", "20250807_mini-v1.7.0_gpt-5-nano",
                 "20250807_mini-v1.7.0_kimi-k2-instruct", "20251209_mini-v1.17.2_devstral-2512",
                 "20251201_mini-v1.17.1_deepseek-v3.2-reasoner"],
    "final": ["20250822_mini-v1.9.1_glm-4.5"],
}
RUN_KEY = re.compile(r"([A-Za-z0-9_.\-]+)/\1\.traj\.json")
STATUS_KEY = re.compile(r"exit_statuses_[0-9.]+\.yaml")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def listing(prefix: str) -> list[tuple[str, int]]:
    """(key, size) of every object under prefix in the public bucket."""
    out, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix} | ({"continuation-token": token} if token else {})
        root = ET.fromstring(get(f"{S3}/?{urllib.parse.urlencode(query)}"))
        out += [(c.findtext("s:Key", namespaces=NS), int(c.findtext("s:Size", namespaces=NS)))
                for c in root.findall("s:Contents", NS)]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s:NextContinuationToken", namespaces=NS)


def download(job: tuple[str, Path, int | None]) -> str:
    url, path, size = job
    if path.exists() and (size is None or path.stat().st_size == size):
        return "kept"
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            data = get(url)
            if size is not None and len(data) != size:
                raise OSError(f"got {len(data)} bytes, the listing says {size}")
            part = path.with_name(path.name + ".part")
            part.write_bytes(data)
            part.replace(path)
            return "new"
        except (OSError, http.client.HTTPException) as err:  # incl. a body cut short
            if attempt == 3:
                raise RuntimeError(f"{url}: {err}") from err
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def jobs_for(entry: str, dest: Path) -> list[tuple[str, Path, int | None]]:
    base = dest / entry
    jobs = [(f"{RAW}/{entry}/{name}", base / name, None) for name in ("metadata.yaml", "per_instance_details.json")]
    prefix = f"bash-only/{entry}/trajs/"
    for key, size in listing(prefix):
        rel = key[len(prefix):]
        if m := RUN_KEY.fullmatch(rel):
            jobs.append((f"{S3}/{urllib.parse.quote(key)}", base / "trajs" / f"{m[1]}.traj.json", size))
        elif STATUS_KEY.fullmatch(rel):
            jobs.append((f"{S3}/{urllib.parse.quote(key)}", base / rel, size))
    return jobs


def main(which: str, dest: str) -> None:
    dest_dir = Path(dest).resolve()
    jobs = [job for entry in ENTRIES[which] for job in jobs_for(entry, dest_dir)]
    print(f"{len(jobs)} files, {sum(j[2] or 0 for j in jobs) / 2**20:.1f} MiB listed", flush=True)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(download, jobs))
    print({r: results.count(r) for r in set(results)})
    for entry in ENTRIES[which]:
        listed = sum(1 for _, path, _ in jobs if path.parent == dest_dir / entry / "trajs")
        runs = sorted((dest_dir / entry / "trajs").glob("*.traj.json"))
        if len(runs) != listed:
            raise SystemExit(f"{entry}: {len(runs)} run files on disk, but the listing has {listed}")
        for path in runs:
            json.loads(path.read_bytes())  # raises if a file is cut short
        print(f"{entry}: {len(runs)} run files, as listed, all parse")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
