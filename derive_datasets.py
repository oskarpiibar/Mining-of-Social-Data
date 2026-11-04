import csv
import json
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "msd_strike.json"
OUT = HERE
DB = HERE / ".derive_datasets.sqlite"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        DROP TABLE IF EXISTS tweets;
        DROP TABLE IF EXISTS retweets;
        DROP TABLE IF EXISTS authors;
        CREATE TABLE tweets (tweet_id TEXT PRIMARY KEY, author_id TEXT);
        CREATE TABLE retweets (
            retweet_id TEXT, source_account_id TEXT, target_tweet_id TEXT,
            created_at TEXT
        );
        CREATE TABLE authors (author_id TEXT PRIMARY KEY, tweet_count INTEGER);
    """)

    originals_path = OUT / "original_tweets.csv"
    with SOURCE.open("r", encoding="utf-8") as src, originals_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=[
            "tweet_id", "author_id", "created_at", "text", "lang",
            "conversation_id", "reply_to_tweet_id", "retweet_count",
            "reply_count", "like_count", "quote_count", "possibly_sensitive",
        ])
        writer.writeheader()
        for line in src:
            if not line.strip():
                continue
            tweet = json.loads(line)
            tweet_id = str(tweet.get("id", ""))
            author_id = str(tweet.get("author_id", ""))
            if tweet_id and author_id:
                db.execute("INSERT OR REPLACE INTO tweets VALUES (?, ?)", (tweet_id, author_id))
                db.execute("""INSERT INTO authors VALUES (?, 1)
                    ON CONFLICT(author_id) DO UPDATE SET tweet_count=tweet_count+1""", (author_id,))

            refs = tweet.get("referenced_tweets") or []
            for ref in refs:
                if ref.get("type") == "retweeted" and tweet_id and author_id:
                    db.execute("INSERT INTO retweets VALUES (?, ?, ?, ?)", (
                        tweet_id, author_id, str(ref.get("id", "")), tweet.get("created_at", "")
                    ))

            # A tweet that quotes another tweet is excluded as well as native retweets.
            if not any(ref.get("type") in {"retweeted", "quoted"} for ref in refs):
                metrics = tweet.get("public_metrics") or {}
                writer.writerow({
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "created_at": tweet.get("created_at", ""),
                    "text": tweet.get("text", ""),
                    "lang": tweet.get("lang", ""),
                    "conversation_id": tweet.get("conversation_id", ""),
                    "reply_to_tweet_id": next((r.get("id", "") for r in refs if r.get("type") == "replied_to"), ""),
                    "retweet_count": metrics.get("retweet_count", ""),
                    "reply_count": metrics.get("reply_count", ""),
                    "like_count": metrics.get("like_count", ""),
                    "quote_count": metrics.get("quote_count", ""),
                    "possibly_sensitive": tweet.get("possibly_sensitive", ""),
                })

            if db.total_changes and db.total_changes % 50000 == 0:
                db.commit()
        db.commit()

    db.execute("CREATE INDEX IF NOT EXISTS retweets_target_idx ON retweets(target_tweet_id)")
    db.commit()

    # Each row represents one directed account-to-account tie, aggregated over tweets.
    with (OUT / "retweet_network_edges.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "source_account_id", "target_account_id", "target_tweet_id",
            "retweet_count", "first_retweeted_at", "last_retweeted_at",
        ])
        query = """
            SELECT r.source_account_id, COALESCE(t.author_id, ''), r.target_tweet_id,
                   COUNT(*), MIN(r.created_at), MAX(r.created_at)
            FROM retweets r LEFT JOIN tweets t ON t.tweet_id = r.target_tweet_id
            GROUP BY r.source_account_id, t.author_id, r.target_tweet_id
            ORDER BY r.source_account_id, t.author_id, r.target_tweet_id
        """
        writer.writerows(db.execute(query))

    with (OUT / "retweet_network_nodes.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["account_id", "tweets_in_source", "retweets_sent", "retweets_received"])
        query = """
            WITH endpoints AS (
                SELECT source_account_id AS account_id FROM retweets
                UNION
                SELECT t.author_id FROM retweets r JOIN tweets t ON t.tweet_id=r.target_tweet_id
            ), sent AS (
                SELECT source_account_id AS account_id, COUNT(*) AS n FROM retweets GROUP BY source_account_id
            ), received AS (
                SELECT t.author_id AS account_id, COUNT(*) AS n
                FROM retweets r JOIN tweets t ON t.tweet_id=r.target_tweet_id GROUP BY t.author_id
            )
            SELECT e.account_id, COALESCE(a.tweet_count, 0), COALESCE(s.n, 0), COALESCE(v.n, 0)
            FROM endpoints e LEFT JOIN authors a USING(account_id)
            LEFT JOIN sent s USING(account_id) LEFT JOIN received v USING(account_id)
            ORDER BY e.account_id
        """
        writer.writerows(db.execute(query))

    db.close()
    DB.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
