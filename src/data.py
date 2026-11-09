"""Load collusion.wiki revisions into one edits table with URLs extracted and masked."""
import gzip, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL_RE = re.compile(r"https?://[^\s\]\)\"'<>|]+")
STRICT_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+)+(?:/[^\s\]\)\"'<>|]*)?|/[\w.-]+/[^\s\]\)\"'<>|]*|\b[0-9a-f]{12,}\b|\b\d{3}-\d{4}\b",
                       re.I)


def norm_url(u: str) -> str:
    u = u.rstrip(".,;:!?*")
    u = re.sub(r"^https?://(www\.)?", "", u.lower())
    return u.rstrip("/")


def load_edits() -> pd.DataFrame:
    rows = [json.loads(l) for l in gzip.open(RAW / "revisions.jsonl.gz", "rt", encoding="utf-8")]
    df = pd.DataFrame(rows)[["rev_id", "page_key", "wiki", "label", "ip16", "time", "body", "body_len", "body_sha256"]]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["body"] = df["body"].fillna("")
    df["label"] = df["label"].fillna("")
    df["urls"] = df["body"].map(lambda b: sorted({norm_url(u) for u in URL_RE.findall(b)}))
    df["domains"] = df["urls"].map(lambda us: sorted({u.split("/")[0] for u in us}))
    # Masked text: every URL replaced, so no detector can see the artifact that defines ground truth.
    df["text_masked"] = df["body"].map(lambda b: URL_RE.sub("[URL]", b))
    # Strict mask: also hide bare domains, hostnames, file names and long ids that could restate the URL.
    df["text_strict"] = df["text_masked"].map(lambda b: STRICT_RE.sub("[TOK]", b))
    # v2 (used by all detectors): URLs deleted outright, so not even the number or position of URLs is visible.
    df["text_nourl"] = df["body"].map(lambda b: re.sub(r"[ \t]{2,}", " ", URL_RE.sub(" ", b)))
    df["text_nourl_strict"] = df["text_nourl"].map(lambda b: re.sub(r"[ \t]{2,}", " ", STRICT_RE.sub(" ", b)))
    df["len_nourl"] = df["text_nourl"].str.len()
    df = df.sort_values(["time", "rev_id"], kind="mergesort").reset_index(drop=True)
    df["eid"] = range(len(df))
    return df


if __name__ == "__main__":
    df = load_edits()
    out = ROOT / "data" / "edits.parquet"
    df.to_parquet(out)
    print(len(df), "edits ->", out)
