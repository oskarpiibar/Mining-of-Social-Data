"""
filters.py - reusable filters for the Climate Strike tweet datasets.

Works on the tables built in lab1.ipynb (all in the project's data/ folder):
  original_tweets.csv         one original tweet per row, incl. a `hashtags` column (big -> read in chunks)
  retweet_events.csv          one row per retweet (who retweeted which tweet, when) - exact, but big
  retweet_network_edges.csv   one row per (retweeting account -> retweeted account) pair - the network
  retweet_network_nodes.csv   one row per account in the retweet network, with attributes

Usage in a notebook (filters.py next to lab1.ipynb):
    from filters import *
    edges = load_edges(EDGES_CSV)
    filter_originals(ORIGINALS_CSV, hashtags=["climatehoax"], start="2019-09-20", end="2019-09-22")
    filter_events(EVENTS_CSV, source="123", target="456", start="2019-09-22", end="2019-09-22")

All timestamps in the data are UTC. Dates are given as 'YYYY-MM-DD' (whole days,
both ends inclusive) or with a time part, e.g. '2019-09-20 14:00' (hour granularity).

Which table to use for network questions:
  * edges  -> structure and totals per account pair (fast, fits in memory).
              One edge row can contain retweets from many days, so time filters on edges
              only tell you when a pair was ACTIVE (see `edges_in_window`).
  * events -> anything that needs exact times or the retweeted tweet: retweets between two
              accounts on a given day/hour, retweets of tweets with a given hashtag.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# `from filters import *` exports only these, so it can't overwrite notebook variables
# (e.g. ORIGINALS_CSV, summary, top_hashtags).
__all__ = [
    "time_mask", "hashtag_mask", "hashtag_column_mask", "hashtags_in",
    "apply_original_filters", "iter_originals", "filter_originals",
    "filter_originals_to_csv", "count_originals",
    "iter_events", "filter_events", "count_events", "retweets_of_tweets", "retweets_by_hashtag",
    "load_edges", "load_nodes", "edges_in_window", "retweets_between",
    "count_retweets_between", "retweets_of_account", "unresolved_edges",
    "pair_weights", "min_weight_pairs", "ego_network",
    "filter_nodes", "edges_among",
]

# data/ sits next to the Lab1/ folder that holds this file
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGINALS_CSV = DATA_DIR / "original_tweets.csv"
EVENTS_CSV = DATA_DIR / "retweet_events.csv"
EDGES_CSV = DATA_DIR / "retweet_network_edges.csv"
NODES_CSV = DATA_DIR / "retweet_network_nodes.csv"

_ORIG_DTYPES = {"tweet_id": str, "author_id": str,
                "conversation_id": str, "reply_to_tweet_id": str}
_EVENT_DTYPES = {"retweet_id": str, "source_account_id": str,
                 "target_account_id": str, "target_tweet_id": str}
_EDGE_DTYPES = {"source_account_id": str, "target_account_id": str,
                "unresolved_target_tweet_id": str}
_NODE_DTYPES = {"account_id": str}


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


def _combine(masks: list, match: str, index) -> pd.Series:
    if not masks:
        return pd.Series(True, index=index)
    out = masks[0]
    for m in masks[1:]:
        out = (out | m) if match == "any" else (out & m)
    return out


def hashtag_mask(text: pd.Series, hashtags, match: str = "any") -> pd.Series:
    """True where the tweet TEXT contains the hashtag(s). Case-insensitive, whole-tag match
    (#climate does NOT match #climatehoax). match='any' (OR) or 'all' (AND).
    Fallback for tables without a `hashtags` column; prefer `hashtag_column_mask`."""
    tags = [t.lstrip("#").casefold() for t in _as_list(hashtags)]
    lowered = text.fillna("").str.casefold()
    masks = [lowered.str.contains(r"(?<!\w)#" + re.escape(t) + r"(?!\w)", regex=True) for t in tags]
    return _combine(masks, match, text.index)


def hashtag_column_mask(tags_col: pd.Series, hashtags, match: str = "any") -> pd.Series:
    """True where the `hashtags` column (space-separated, lower-cased, taken from the JSON `entities`)
    contains the hashtag(s). match='any' (OR) or 'all' (AND)."""
    wanted = [t.lstrip("#").casefold() for t in _as_list(hashtags)]
    padded = " " + tags_col.fillna("").astype(str) + " "
    masks = [padded.str.contains(" " + t + " ", regex=False) for t in wanted]
    return _combine(masks, match, tags_col.index)


def hashtags_in(df: pd.DataFrame, hashtags, match: str = "any") -> pd.Series:
    """Hashtag mask for an original-tweets DataFrame: uses the `hashtags` column when the table has one,
    otherwise searches the text."""
    if "hashtags" in df.columns:
        return hashtag_column_mask(df["hashtags"], hashtags, match)
    return hashtag_mask(df["text"], hashtags, match)


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
        m &= hashtags_in(df, hashtags, match)
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


# ----------------------------------------------------------------------------
# 2) RETWEET EVENTS  (one row per retweet; exact times; chunked)
# ----------------------------------------------------------------------------
def apply_event_filters(
    df: pd.DataFrame,
    source=None, target=None, both_directions: bool = False,
    start=None, end=None,
    target_tweet_ids=None,
) -> pd.DataFrame:
    """Criteria (AND). source = retweeting account(s); target = retweeted account(s).
    With both_directions=True and both source and target given, retweets in either direction match.
    target_tweet_ids = only retweets of these tweets."""
    m = time_mask(df["created_at"], start, end)
    if source is not None and target is not None and both_directions:
        s, t = _as_list(source), _as_list(target)
        m &= ((df["source_account_id"].isin(s) & df["target_account_id"].isin(t))
              | (df["source_account_id"].isin(t) & df["target_account_id"].isin(s)))
    else:
        if source is not None:
            m &= df["source_account_id"].isin(_as_list(source))
        if target is not None:
            m &= df["target_account_id"].isin(_as_list(target))
    if target_tweet_ids is not None:
        m &= df["target_tweet_id"].isin(set(_as_list(target_tweet_ids)))
    return df[m]


def iter_events(path: Path = EVENTS_CSV, chunksize: int = 500_000, **criteria) -> Iterator[pd.DataFrame]:
    """Yield filtered chunks of retweet_events.csv (memory-friendly)."""
    for chunk in pd.read_csv(path, dtype=_EVENT_DTYPES, chunksize=chunksize):
        out = apply_event_filters(chunk, **criteria)
        if len(out):
            yield out


def filter_events(path: Path = EVENTS_CSV, **criteria) -> pd.DataFrame:
    """Retweets matching the criteria, one row each (retweet_id, created_at, source_account_id,
    target_account_id, target_tweet_id, target_from_mention). Times are exact."""
    parts = list(iter_events(path, **criteria))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path, dtype=_EVENT_DTYPES, nrows=0)


def count_events(path: Path = EVENTS_CSV, **criteria) -> int:
    """Number of retweets matching the criteria (e.g. A retweeting B on one day)."""
    return sum(len(p) for p in iter_events(path, **criteria))


def retweets_of_tweets(tweet_ids, path: Path = EVENTS_CSV, **criteria) -> pd.DataFrame:
    """Retweets of the given original tweet ids."""
    return filter_events(path, target_tweet_ids=tweet_ids, **criteria)


def retweets_by_hashtag(hashtags, match: str = "any", originals_path: Path = ORIGINALS_CSV,
                        events_path: Path = EVENTS_CSV, **criteria) -> pd.DataFrame:
    """Retweets of original tweets that contain the hashtag(s).
    start/end (in **criteria) limit WHEN the retweet happened. Only retweets of tweets that are in
    original_tweets.csv can be matched (the hashtags of other retweeted tweets are unknown)."""
    ids = filter_originals(originals_path, hashtags=hashtags, match=match)["tweet_id"]
    return retweets_of_tweets(ids, events_path, **criteria)


# ----------------------------------------------------------------------------
# 3) RETWEET NETWORK  (edges fit in memory: load once, then filter)
# ----------------------------------------------------------------------------
def load_edges(path: Path = EDGES_CSV) -> pd.DataFrame:
    """Columns: source_account_id, target_account_id, unresolved_target_tweet_id, retweet_count,
    distinct_tweets_retweeted, retweets_target_from_mention, first_retweeted_at, last_retweeted_at.
    One row per (retweeter -> retweeted account). target_account_id is NaN only when the author of the
    retweeted tweet could not be determined; those rows are per tweet and carry unresolved_target_tweet_id."""
    return pd.read_csv(path, dtype=_EDGE_DTYPES)


def load_nodes(path: Path = NODES_CSV) -> pd.DataFrame:
    """Account attributes: tweets_in_source, first_seen, last_seen, retweets_sent, retweets_received and
    (after the enrichment step in the notebook) original_tweets, likes_on_originals, ..."""
    return pd.read_csv(path, dtype=_NODE_DTYPES)


def edges_in_window(edges: pd.DataFrame, start=None, end=None, mode: str = "overlap") -> pd.DataFrame:
    """Edges active in [start, end] (UTC; dates are whole days).
    mode='overlap'   : edge has at least one retweet in the window (first <= end and last >= start).
                       retweet_count is still the pair's TOTAL, so it can include retweets outside the window.
    mode='contained' : all of the edge's retweets are inside the window, so retweet_count is exact for the window
                       (pairs whose retweets spread beyond the window are left out).
    For exact counts per day/hour use `filter_events` / `count_events` instead."""
    if mode == "contained":
        m = time_mask(edges["first_retweeted_at"], start=start) & time_mask(edges["last_retweeted_at"], end=end)
    elif mode == "overlap":
        m = time_mask(edges["first_retweeted_at"], end=end) & time_mask(edges["last_retweeted_at"], start=start)
    else:
        raise ValueError("mode must be 'overlap' or 'contained'")
    return edges[m]


def retweets_between(edges: pd.DataFrame, source, target, start=None, end=None,
                     both_directions: bool = False, mode: str = "overlap") -> pd.DataFrame:
    """Edge row(s) where account `source` retweeted account `target` (optionally also target -> source).
    With start/end see `edges_in_window` for what the window means."""
    s, t = str(source), str(target)
    m = (edges["source_account_id"] == s) & (edges["target_account_id"] == t)
    if both_directions:
        m |= (edges["source_account_id"] == t) & (edges["target_account_id"] == s)
    out = edges[m]
    if start is not None or end is not None:
        out = edges_in_window(out, start, end, mode)
    return out


def count_retweets_between(edges: pd.DataFrame, source, target, **kw) -> int:
    """Total number of times `source` retweeted `target` (sum of retweet_count)."""
    return int(retweets_between(edges, source, target, **kw)["retweet_count"].sum())


def retweets_of_account(edges: pd.DataFrame, account, role: str = "both",
                        start=None, end=None, mode: str = "overlap") -> pd.DataFrame:
    """role='sent'     -> edges of retweets made BY the account
       role='received' -> edges of retweets OF the account's tweets
       role='both'     -> either"""
    a = str(account)
    sent = edges["source_account_id"] == a
    received = edges["target_account_id"] == a
    out = edges[{"sent": sent, "received": received, "both": sent | received}[role]]
    if start is not None or end is not None:
        out = edges_in_window(out, start, end, mode)
    return out


def unresolved_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Retweets whose original author is unknown (not in the dataset and no usable 'RT @user' mention)."""
    return edges[edges["target_account_id"].isna()]


def pair_weights(edges: pd.DataFrame, known_targets_only: bool = True) -> pd.DataFrame:
    """Weighted account->account network: weight = number of retweets, n_tweets = distinct tweets retweeted.
    (Rows are already per pair; this also merges any duplicates and sorts by weight.)"""
    e = edges.dropna(subset=["target_account_id"]) if known_targets_only else edges
    return (e.groupby(["source_account_id", "target_account_id"], as_index=False, dropna=False)
             .agg(weight=("retweet_count", "sum"),
                  n_tweets=("distinct_tweets_retweeted", "sum"),
                  first_retweeted_at=("first_retweeted_at", "min"),
                  last_retweeted_at=("last_retweeted_at", "max"))
             .sort_values("weight", ascending=False, ignore_index=True))


def min_weight_pairs(edges: pd.DataFrame, k: int) -> pd.DataFrame:
    """Account pairs where the source retweeted the target at least k times."""
    pw = pair_weights(edges)
    return pw[pw["weight"] >= k].reset_index(drop=True)


def ego_network(edges: pd.DataFrame, account, hops: int = 1) -> pd.DataFrame:
    """All edges within `hops` steps of an account (either direction)."""
    nodes = {str(account)}
    for _ in range(hops):
        m = edges["source_account_id"].isin(nodes) | edges["target_account_id"].isin(nodes)
        sub = edges[m]
        nodes |= set(sub["source_account_id"]) | set(sub["target_account_id"].dropna())
    return edges[edges["source_account_id"].isin(nodes) & edges["target_account_id"].isin(nodes)]


# ----------------------------------------------------------------------------
# 4) NODES
# ----------------------------------------------------------------------------
def filter_nodes(nodes: pd.DataFrame, min_sent: int = 0, min_received: int = 0,
                 min_tweets: int = 0, accounts=None) -> pd.DataFrame:
    """Accounts with at least the given activity; optionally restricted to a list of account ids."""
    m = ((nodes["retweets_sent"] >= min_sent) & (nodes["retweets_received"] >= min_received)
         & (nodes["tweets_in_source"] >= min_tweets))
    if accounts is not None:
        m &= nodes["account_id"].isin(_as_list(accounts))
    return nodes[m]


def edges_among(edges: pd.DataFrame, accounts) -> pd.DataFrame:
    """Sub-network: only edges where both ends are in `accounts` (e.g. the output of filter_nodes)."""
    ids = set(_as_list(accounts))
    return edges[edges["source_account_id"].isin(ids) & edges["target_account_id"].isin(ids)]