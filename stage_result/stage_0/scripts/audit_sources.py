"""Record primary-source identities and actual public repository contents.

This audit never installs, clones, executes or reproduces external methods.
The manifest preserves access failures and hashes, rather than treating a link
or a repository README as proof of runnable code.
"""
import concurrent.futures
import hashlib
import json
import re
from html import unescape
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1]
PAPERS = {
    "revising": "https://arxiv.org/abs/2404.06109",
    "fastgs": "https://arxiv.org/abs/2511.04283",
    "salientgs": "https://arxiv.org/abs/2607.11285",
    "energygs": "https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Energy-GS_Image_Energy-guided_Pose_Alignment_Gaussian_Splatting_with_redesigned_pose_CVPR_2026_paper.html",
    "bsgs": "https://arxiv.org/abs/2510.12493",
    "harmogs": "https://arxiv.org/abs/2605.13073",
    "blitzgs": "https://arxiv.org/abs/2605.13794",
    "legs": "https://arxiv.org/abs/2605.00408",
}
REPOS = ["fastgs/FastGS", "Six-Bit-TX/SalientGS", "SkylerGao/ENGS", "wsxujm/bsgs", "AkierRaee/BlitzGS", "AaronNZH/LeGS"]


def get(url):
    response = requests.get(url, timeout=35, headers={"User-Agent": "CCD-GS-Stage0-ReadOnlyAudit/1.0"})
    response.raise_for_status()
    return response


def paper(item):
    key, url = item
    try:
        r = get(url)
        # Metadata only; no article text is redistributed.
        title = re.search(r'<meta name="citation_title" content="([^"]+)"', r.text)
        authors = re.findall(r'<meta name="citation_author" content="([^"]+)"', r.text)
        identifier = re.search(r'<meta name="citation_arxiv_id" content="([^"]+)"', r.text)
        history = re.search(r'<div class="submission-history">(.*?)</div>', r.text, re.S)
        versions = []
        if history:
            plain = unescape(re.sub(r"<[^>]+>", " ", history.group(1)))
            versions = re.findall(r"\[v(\d+)\]\s*([^\(\n]+)", plain)
        return {"id": key, "url": url, "status": r.status_code,
            "citation_title": unescape(title.group(1)) if title else None,
            "authors": [unescape(x) for x in authors],
            "citation_arxiv_id": identifier.group(1) if identifier else None,
            "version_history": versions, "response_sha256": hashlib.sha256(r.content).hexdigest()}
    except Exception as exc:
        return {"id": key, "url": url, "error": str(exc), "verified": False}


def repo(name):
    try:
        root = get("https://api.github.com/repos/" + name).json()
        branch = root["default_branch"]
        commit = get(f"https://api.github.com/repos/{name}/commits/{branch}").json()
        sha = commit["sha"]
        tree = get(f"https://api.github.com/repos/{name}/git/trees/{sha}?recursive=1").json()
        entries = tree.get("tree", [])
        files = [e["path"] for e in entries if e["type"] == "blob"]
        code = [p for p in files if Path(p).suffix.lower() in {".py", ".cu", ".cpp", ".c", ".h", ".hpp", ".sh"}]
        license_data = root.get("license") or {}
        return {"repository": name, "url": root["html_url"], "default_branch": branch,
            "head_sha": sha, "head_date": commit["commit"]["committer"]["date"],
            "license_spdx": license_data.get("spdx_id"), "tree_truncated": tree.get("truncated"),
            "file_count": len(files), "code_file_count": len(code), "files": files,
            "code_present": bool(code), "runnability_verified": False,
            "audit_level": "recursive file inventory and pinned commit; no execution"}
    except Exception as exc:
        return {"repository": name, "error": str(exc), "runnability_verified": False}


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        papers = list(pool.map(paper, PAPERS.items()))
        repos = list(pool.map(repo, REPOS))
    result = {"verification_date_local": "2026-10-10", "timezone": "Asia/Saigon",
        "papers": papers, "repositories": repos,
        "caution": "file presence is not execution evidence; lack of code refers to the checked public default-branch tree"}
    dest = OUT / "evidence" / "source_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"papers": [{k: p.get(k) for k in ["id", "citation_arxiv_id", "version_history", "error"]} for p in papers],
        "repos": [{k: r.get(k) for k in ["repository", "head_sha", "file_count", "code_file_count", "error"]} for r in repos]}, indent=2))
