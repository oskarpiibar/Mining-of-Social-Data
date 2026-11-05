"""
filters.py - reusable filters for the Climate Strike tweet datasets.

Works on the three tables built in lab1.ipynb:
  data/original_tweets.csv          one original tweet per row (big -> read in chunks)
  data/retweet_network_edges.csv    one row per (retweeter, retweeted tweet), small enough for RAM
  data/retweet_network_nodes.csv    per-account summary (only after the nodes bug is fixed)

Usage in a notebook:
    from filters import *
    edges = load_edges()
    filter_originals(hashtags=["climatehoax"], start="2019-09-20", end="2019-09-22")
    retweets_between(edges, "123", "456")

All timestamps in the data are UTC. Dates are given as 'YYYY-MM-DD' (whole days,
both ends inclusive) or with a time part, e.g. '2019-09-20 14:00' (hour granularity).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

import pandas as pd

DATA_DIR = Path("data")
ORIGINALS_CSV = DATA_DIR / "original_tweets.csv"
EDGES_CSV = DATA_DIR / "retweet_network_edges.csv"
NODES_CSV = DATA_DIR / "retweet_network_nodes.csv"

_ORIG_DTYPES = {"tweet_id": str, "author_id": str,
                "conversation_id": str, "reply_to_tweet_id": str}
_EDGE_DTYPES = {"source_account_id": str, "target_account_id": str,
                "target_tweet_id": str}

_HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _norm(x) -> str:
    """'2019-09-20 14:00' / date / Timestamp -> '2019-09-20T14:00' (prefix of the ISO strings in the data)."""
    return str(x).strip().replace(" ", "T")


def time_mask(ts: pd.Series, start=None, end=None) -> pd.Series:
    """Boolean mask for ISO timestamp strings, comparing only as many characters as the bound has.
    start/end are inclusive at the granularity given (a date covers the whole day)."""
    mask = pd.Series(True, index=ts.index)
    ts = ts.fillna("")
    if start is not None:
        s = _norm(start)
        mask &= ts.str.slice(0, len(s)) >= s
    if end is not None:
        e = _norm(end)
        mask &= ts.str.slice(0, len(e)) <= e
    return mask


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return [str(x)]
    return [str(i) for i in x]


def hashtag_mask(text: pd.Series, hashtags, match: str = "any") -> pd.Series:
    """True where the tweet text contains the hashtag(s). Case-insensitive, whole-tag match
    (#climate does NOT match #climatehoax). match='any' (OR) or 'all' (AND)."""
    tags = [t.lstrip("#").lower() for t in _as_list(hashtags)]
    lowered = text.fillna("").str.lower()
    masks = [lowered.str.contains(r"(?<!\w)#" + re.escape(t) + r"(?!\w)", regex=True) for t in tags]
    if not masks:
        return pd.Series(True, index=text.index)
    out = masks[0]
    for m in masks[1:]:
        out = (out | m) if match == "any" else (out & m)
    return out


# ----------------------------------------------------------------------------
# 1) ORIGINAL TWEETS  (chunked, so it also works when the file is large)
# ----------------------------------------------------------------------------
def apply_original_filters(
    df: pd.DataFrame,
    start=None, end=None,
    hashtags=None, match: str = "any",
    text_regex: Optional[str] = None,
    lang=None,
    authors=None,
    exclude_authors=None,
    replies: str = "keep",          # 'keep' | 'exclude' | 'only'
    min_likes: Optional[int] = None,
    min_retweets: Optional[int] = None,
    min_replies: Optional[int] = None,
    min_quotes: Optional[int] = None,
) -> pd.DataFrame:
    """Apply all given criteria (AND) to a DataFrame shaped like original_tweets.csv."""
    m = time_mask(df["created_at"], start, end)
    if hashtags:
        m &= hashtag_mask(df["text"], hashtags, match)
    if text_regex:
        m &= df["text"].fillna("").str.contains(text_regex, case=False, regex=True)
    if lang:
        m &= df["lang"].isin(_as_list(lang))
    if authors is not None:
        m &= df["author_id"].isin(_as_list(authors))
    if exclude_authors is not None:
        m &= ~df["author_id"].isin(_as_list(exclude_authors))
    if replies == "exclude":
        m &= df["reply_to_tweet_id"].isna()
    elif replies == "only":
        m &= df["reply_to_tweet_id"].notna()
    for col, thr in (("like_count", min_likes), ("retweet_count", min_retweets),
                     ("reply_count", min_replies), ("quote_count", min_quotes)):
        if thr is not None:
            m &= df[col].fillna(0) >= thr
    return df[m]


def iter_originals(path: Path = ORIGINALS_CSV, chunksize: int = 200_000, **criteria) -> Iterator[pd.DataFrame]:
    """Yield filtered chunks of original_tweets.csv (memory-friendly)."""
    for chunk in pd.read_csv(path, dtype=_ORIG_DTYPES, chunksize=chunksize):
        out = apply_original_filters(chunk, **criteria)
        if len(out):
            yield out


def filter_originals(path: Path = ORIGINALS_CSV, **criteria) -> pd.DataFrame:
    """Return all original tweets matching the criteria as one DataFrame."""
    parts = list(iter_originals(path, **criteria))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path, dtype=_ORIG_DTYPES, nrows=0)


def filter_originals_to_csv(out_path, path: Path = ORIGINALS_CSV, **criteria) -> int:
    """Stream the matching tweets to a new CSV (for results too big for RAM). Returns the row count."""
    n, first = 0, True
    for part in iter_originals(path, **criteria):
        part.to_csv(out_path, mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig")
        n, first = n + len(part), False
    return n


def count_originals(path: Path = ORIGINALS_CSV, **criteria) -> int:
    return sum(len(p) for p in iter_originals(path, **criteria))


def originals_per_day(path: Path = ORIGINALS_CSV, **criteria) -> pd.Series:
    """Number of matching original tweets per UTC day."""
    c = Counter()
    for part in iter_originals(path, **criteria):
        c.update(part["created_at"].str.slice(0, 10))
    return pd.Series(c, dtype="int64").sort_index().rename("tweets")


def originals_per_hour(path: Path = ORIGINALS_CSV, **criteria) -> pd.Series:
    c = Counter()
    for part in iter_originals(path, **criteria):
        c.update(part["created_at"].str.slice(0, 13))
    return pd.Series(c, dtype="int64").sort_index().rename("tweets")


def top_hashtags(n: int = 20, path: Path = ORIGINALS_CSV, **criteria) -> pd.Series:
    """Most common hashtags (lower-cased) among the matching original tweets."""
    c = Counter()
    for part in iter_originals(path, **criteria):
        for t in part["text"].fillna(""):
            c.update(h.lower() for h in _HASHTAG_RE.findall(t))
    return pd.Series(dict(c.most_common(n)), dtype="int64", name="count")


# ----------------------------------------------------------------------------
# 2) RETWEET NETWORK  (edges fit in memory: load once, then filter)
# ----------------------------------------------------------------------------
def load_edges(path: Path = EDGES_CSV) -> pd.DataFrame:
    """Edge table: one row per (retweeting account, retweeted tweet).
    target_account_id is NaN when the retweeted tweet is not itself in the source file."""
    return pd.read_csv(path, dtype=_EDGE_DTYPES)


def retweets_between(edges: pd.DataFrame, source, target, start=None, end=None,
                     both_directions: bool = False) -> pd.DataFrame:
    """Edge rows where account `source` retweeted account `target` (optionally also target -> source)."""
    s, t = str(source), str(target)
    m = (edges["source_account_id"] == s) & (edges["target_account_id"] == t)
    if both_directions:
        m |= (edges["source_account_id"] == t) & (edges["target_account_id"] == s)
    m &= time_mask(edges["first_retweeted_at"], start, end)
    return edges[m]


def count_retweets_between(edges: pd.DataFrame, source, target, **kw) -> int:
    """Total number of times `source` retweeted `target` (sum of retweet_count)."""
    return int(retweets_between(edges, source, target, **kw)["retweet_count"].sum())


def retweets_of_account(edges: pd.DataFrame, account, role: str = "both", start=None, end=None) -> pd.DataFrame:
    """role='sent'     -> retweets made BY the account
       role='received' -> retweets OF the account's tweets
       role='both'     -> either"""
    a = str(account)
    sent = edges["source_account_id"] == a
    received = edges["target_account_id"] == a
    m = {"sent": sent, "received": received, "both": sent | received}[role]
    m &= time_mask(edges["first_retweeted_at"], start, end)
    return edges[m]


def edges_by_day(edges: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Edges whose (first) retweet falls in [start, end]. Note: if an edge has retweet_count > 1 and its
    retweets span the window boundary, retweet_count still counts all of them."""
    return edges[time_mask(edges["first_retweeted_at"], start, end)]


def retweets_per_day(edges: pd.DataFrame) -> pd.Series:
    return (edges.groupby(edges["first_retweeted_at"].str.slice(0, 10))["retweet_count"]
                 .sum().rename("retweets"))


def retweets_per_hour(edges: pd.DataFrame) -> pd.Series:
    return (edges.groupby(edges["first_retweeted_at"].str.slice(0, 13))["retweet_count"]
                 .sum().rename("retweets"))


def retweets_of_tweets(edges: pd.DataFrame, tweet_ids: Iterable) -> pd.DataFrame:
    """Edges that retweet any of the given tweet ids."""
    return edges[edges["target_tweet_id"].isin({str(i) for i in tweet_ids})]


def retweets_by_hashtag(edges: pd.DataFrame, hashtags, match: str = "any",
                        originals_path: Path = ORIGINALS_CSV, **criteria) -> pd.DataFrame:
    """Retweets of original tweets that contain the hashtag(s).
    Limitation: hashtags are only known for tweets that are themselves in original_tweets.csv, so
    retweets whose source tweet is outside the dataset (target_account_id NaN) cannot be matched."""
    ids = filter_originals(originals_path, hashtags=hashtags, match=match, **criteria)["tweet_id"]
    return retweets_of_tweets(edges, ids)


def pair_weights(edges: pd.DataFrame, known_targets_only: bool = True) -> pd.DataFrame:
    """Collapse tweet-level edges to account->account edges (the usual weighted retweet network)."""
    e = edges.dropna(subset=["target_account_id"]) if known_targets_only else edges
    return (e.groupby(["source_account_id", "target_account_id"], as_index=False, dropna=False)
             .agg(weight=("retweet_count", "sum"),
                  n_tweets=("target_tweet_id", "nunique"),
                  first_retweeted_at=("first_retweeted_at", "min"),
                  last_retweeted_at=("last_retweeted_at", "max"))
             .sort_values("weight", ascending=False, ignore_index=True))


def top_pairs(edges: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Account pairs with the most retweets between them."""
    return pair_weights(edges).head(n)


def min_weight_pairs(edges: pd.DataFrame, k: int) -> pd.DataFrame:
    """Account pairs where the source retweeted the target at least k times."""
    pw = pair_weights(edges)
    return pw[pw["weight"] >= k].reset_index(drop=True)


def top_retweeters(edges: pd.DataFrame, n: int = 20) -> pd.Series:
    return edges.groupby("source_account_id")["retweet_count"].sum().nlargest(n).rename("retweets_sent")


def top_retweeted(edges: pd.DataFrame, n: int = 20) -> pd.Series:
    return edges.groupby("target_account_id")["retweet_count"].sum().nlargest(n).rename("retweets_received")


def ego_network(edges: pd.DataFrame, account, hops: int = 1) -> pd.DataFrame:
    """All edges within `hops` steps of an account (either direction)."""
    nodes = {str(account)}
    for _ in range(hops):
        m = edges["source_account_id"].isin(nodes) | edges["target_account_id"].isin(nodes)
        sub = edges[m]
        nodes |= set(sub["source_account_id"]) | set(sub["target_account_id"].dropna())
    return edges[edges["source_account_id"].isin(nodes) & edges["target_account_id"].isin(nodes)]