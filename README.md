# AI Daily News -> Slack Digest

*Last updated: October 3, 2025*

> **Note**: The GitHub Actions workflow is scheduled to run daily at 8 AM PT (15:00/16:00 UTC)

A small Python script that fetches targeted, motivational AI news and learning resources from curated feeds and posts a short digest to Slack via your existing bot or webhook.

## Features
- Curated AI sources (OpenAI, DeepMind, Google, Anthropic, HF, AWS, Azure, etc.)
- Filters for practical learning value (tutorials, guides, case studies) and product launches
- Scores and ranks items; configurable window, min score, and max items
- Posts as rich Slack blocks using your existing bot token or Incoming Webhook

## Quick Start

1) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment
```bash
cp .env.example .env
# Edit .env to set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID
# or set SLACK_WEBHOOK_URL
```

3) Run once
```bash
python ai_news_digest.py
```

If successful, you should see a digest posted to your target Slack channel.

## Configuration
- `.env`
  - `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` OR `SLACK_WEBHOOK_URL`
  - `MAX_ARTICLES` (default 8)
  - `MIN_SCORE` (default 0.6)
  - `TIME_WINDOW_HOURS` (default 72)
  - `NEWS_SOURCES` (comma-separated optional override)
  - `KEYWORDS` (comma-separated optional override)

## Scheduling (cron)
To post every weekday at 9:00 AM:
```bash
crontab -e
```
Add:
```
0 9 * * 1-5 /bin/bash -lc 'cd "$HOME/AI Daily News" && source .venv/bin/activate && python ai_news_digest.py >> digest.log 2>&1'
```

## Tips
- Invite your Slack bot to the target channel.
- If using a webhook, no channel ID is needed but the webhook must be connected to the right channel.
- Tune `MIN_SCORE` upwards (e.g., 0.7–0.8) for stricter relevance.
- Add or remove sources via `NEWS_SOURCES` to match your team’s focus.

## Notes
This script scrapes article text heuristically when necessary to score relevance. Respect robots.txt and site terms. If a source blocks scraping, the script will fall back to feed text.
