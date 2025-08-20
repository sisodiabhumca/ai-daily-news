#!/usr/bin/env python3
import os
import re
import time
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import argparse

import feedparser
import requests
from bs4 import BeautifulSoup
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_SOURCES = [
    # Major AI research and product blogs
    "https://openai.com/blog/rss.xml",
    "https://deepmind.google/discover/rss/",
    "https://ai.googleblog.com/feeds/posts/default",
    "https://www.anthropic.com/news/rss.xml",
    "https://machinelearning.apple.com/rss.xml",
    "https://www.microsoft.com/en-us/research/feed/",
    "https://developer.nvidia.com/blog/category/ai-data-science/feed/",
    "https://ai.facebook.com/blog/rss/",
    "https://engineering.atspotify.com/feed/",
    "https://netflixtechblog.com/feed",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://azure.microsoft.com/en-us/blog/topics/ai-machine-learning/feed/",
    "https://cloud.google.com/blog/topics/ai-ml/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://blog.langchain.dev/rss/",
    "https://blog.llamaindex.ai/feed",
    "https://blog.eleuther.ai/rss/",
    "https://cohere.com/blog/rss.xml",
    "https://stability.ai/blog/rss.xml",
    # Industry news
    "https://www.marktechpost.com/category/ai/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theverge.com/ai-artificial-intelligence/rss",
    "https://arstechnica.com/information-technology/feed/",
    "https://www.databricks.com/blog/feed",
    "https://www.snowflake.com/blog/feed/",
    "http://export.arxiv.org/rss/cs.AI",
    "http://export.arxiv.org/rss/cs.CL",
    "https://www.semianalysis.com/feed",
    "https://www.lesswrong.com/feed.xml",
]

DEFAULT_KEYWORDS = [
    # Learning-oriented
    "how", "tutorial", "guide", "hands-on", "implement", "walkthrough",
    "best practices", "case study", "playbook", "examples",
    # Product and GA signals
    "launch", "released", "general availability", "preview", "announce",
    "roadmap", "update", "capability", "integration",
    # Core AI topics
    "llm", "gpt", "transformer", "rag", "retrieval", "vector",
    "fine-tune", "fine tuning", "distillation", "inference", "prompt",
    "agent", "multi-agent", "evaluation", "benchmark",
    # Enterprise use
    "productivity", "workflow", "automation", "customer", "support", "sales",
    "engineering", "analytics", "governance", "security", "compliance",
    # Companies/tech frequently relevant
    "openai", "anthropic", "google", "deepmind", "microsoft", "azure",
    "aws", "bedrock", "meta", "nvidia", "snowflake", "databricks",
]

NEGATIVE_SIGNALS = [
    "opinion", "rumor", "weekly", "roundup", "recap", "sponsored",
    "meme", "satire", "fiction",
]

MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "8"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.6"))
TIME_WINDOW_HOURS = int(os.getenv("TIME_WINDOW_HOURS", "24"))

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")
# Alternative webhook support (optional)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

CUSTOM_SOURCES = [s.strip() for s in os.getenv("NEWS_SOURCES", "").split(",") if s.strip()]
CUSTOM_KEYWORDS = [k.strip().lower() for k in os.getenv("KEYWORDS", "").split(",") if k.strip()]

SOURCES = CUSTOM_SOURCES or DEFAULT_SOURCES
KEYWORDS = CUSTOM_KEYWORDS or DEFAULT_KEYWORDS

session = requests.Session()
session.headers.update({
    "User-Agent": "AI-Daily-News-Bot/1.0 (+https://example.com)"
})

# -----------------------------
# Helpers
# -----------------------------

def strip_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text or "", "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return text or ""


def fetch_article_text(url: str, timeout: int = 10) -> str:
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Heuristic: grab visible text from article containers
        candidates = []
        for selector in [
            "article",
            "main",
            "div.post", "div.article", "div.entry-content",
            "section.post", "section.article",
        ]:
            for el in soup.select(selector):
                text = el.get_text(" ", strip=True)
                if text and len(text) > 300:
                    candidates.append(text)
        if candidates:
            return max(candidates, key=len)[:15000]
        # Fallback to full page text
        return soup.get_text(" ", strip=True)[:15000]
    except Exception:
        return ""


def score_item(title: str, summary: str, content: str) -> float:
    text = f"{title}\n{summary}\n{content}".lower()

    # Keyword hits
    hits = sum(1 for kw in KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text))
    density = hits / max(1, len(text) / 1000)  # per 1k chars

    # Signals of practical value
    practical = 0
    for token in ["how", "tutorial", "guide", "case study", "example", "implement"]:
        if token in text:
            practical += 1

    # Penalize negative signals
    penalties = sum(1 for n in NEGATIVE_SIGNALS if n in text)

    # Content length heuristic
    length_score = min(len(content) / 3000.0, 1.0)  # cap at ~3k chars

    score = 0.45 * density + 0.35 * (practical / 3) + 0.2 * length_score - 0.15 * penalties
    return max(0.0, min(1.0, score))


def within_time_window(published: datetime) -> bool:
    # Strict: only include items with a known timestamp within the window
    if not published:
        return False
    now = datetime.now(timezone.utc)
    return (now - published) <= timedelta(hours=TIME_WINDOW_HOURS)


def parse_time(entry: Dict[str, Any]) -> datetime | None:
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime.fromtimestamp(time.mktime(entry.updated_parsed), tz=timezone.utc)
    except Exception:
        return None
    return None


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]


# -----------------------------
# Core pipeline
# -----------------------------

def fetch_entries(sources: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for src in sources:
        try:
            logging.info(f"Fetching feed: {src}")
            feed = feedparser.parse(src)
            for e in feed.entries:
                title = strip_html(getattr(e, "title", "").strip())
                link = getattr(e, "link", "").strip()
                summary = strip_html(getattr(e, "summary", getattr(e, "description", "")))
                published = parse_time(e)
                if not link or not title:
                    continue
                items.append({
                    "source": feed.feed.get("title", src),
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                })
        except Exception as ex:
            logging.warning(f"Failed to fetch {src}: {ex}")
    return items


def enrich_and_score(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    enriched: List[Dict[str, Any]] = []

    for it in items:
        if not within_time_window(it.get("published")):
            continue
        url = it["link"]
        sig = hash_url(url)
        if sig in seen:
            continue
        seen.add(sig)

        content = fetch_article_text(url)
        score = score_item(it["title"], it.get("summary", ""), content)
        if score < MIN_SCORE:
            continue
        it.update({
            "content": content,
            "score": score,
        })
        enriched.append(it)
    # Sort by score desc, then recency desc
    enriched.sort(key=lambda x: (x["score"], x.get("published") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return enriched[:MAX_ARTICLES]


# -----------------------------
# Slack formatting and posting
# -----------------------------

def build_slack_blocks(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    header_text = f"AI Learning Digest — {datetime.now().strftime('%b %d, %Y')}"
    blocks.extend([
        {"type": "header", "text": {"type": "plain_text", "text": header_text}},
        {"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"Curated sources: {len(SOURCES)} | Window: {TIME_WINDOW_HOURS}h | Min score: {MIN_SCORE}"}
        ]},
        {"type": "divider"},
    ])

    if not articles:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "No highly relevant AI items found in the selected window."}})
        return blocks

    for idx, a in enumerate(articles, start=1):
        pub = a.get("published")
        when = pub.astimezone().strftime("%b %d %H:%M") if isinstance(pub, datetime) else "recent"
        snippet = (a.get("summary") or a.get("content", "")).strip()
        snippet = snippet[:280] + ("…" if len(snippet) > 280 else "")
        score_str = f"{a['score']:.2f}"
        section = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{idx}. <{a['link']}|{a['title']}>*\n_{a['source']}_ • {when} • score {score_str}\n{snippet}",
            },
        }
        blocks.append(section)
        blocks.append({"type": "divider"})

    footer = {
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "Tip: Save items that look like tutorials, case studies, or product launches most relevant to your team."}
        ],
    }
    blocks.append(footer)
    return blocks


def post_to_slack(blocks: List[Dict[str, Any]]):
    if SLACK_WEBHOOK_URL:
        try:
            resp = session.post(SLACK_WEBHOOK_URL, json={"blocks": blocks, "text": "AI Learning Digest"}, timeout=10)
            resp.raise_for_status()
            logging.info("Posted digest to Slack via webhook")
            return
        except Exception as ex:
            logging.error(f"Webhook post failed: {ex}")
            # fall through to bot token if available

    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        raise RuntimeError("Missing Slack configuration. Provide SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN and SLACK_CHANNEL_ID")

    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        client.chat_postMessage(channel=SLACK_CHANNEL_ID, text="AI Learning Digest", blocks=blocks, unfurl_links=False, unfurl_media=False)
        logging.info("Posted digest to Slack via bot token")
    except SlackApiError as e:
        logging.error(f"Slack API error: {e.response['error']}")
        raise


# -----------------------------
# Main
# -----------------------------

def run(dry_run: bool = False) -> int:
    logging.info(f"Starting AI news digest. Sources={len(SOURCES)}, Window={TIME_WINDOW_HOURS}h")
    items = fetch_entries(SOURCES)
    logging.info(f"Fetched {len(items)} raw items")
    curated = enrich_and_score(items)
    logging.info(f"Selected {len(curated)} items after scoring")
    blocks = build_slack_blocks(curated)
    if dry_run:
        print(json.dumps({"preview": True, "count": len(curated), "blocks": blocks}, ensure_ascii=False, indent=2))
    else:
        post_to_slack(blocks)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Learning Digest to Slack")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and score, print blocks JSON without posting to Slack")
    args = parser.parse_args()
    raise SystemExit(run(dry_run=args.dry_run))
