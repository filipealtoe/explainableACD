"""
Extract Factual Claims from Viral Anomalies

This script:
1. Loads viral anomaly events (predicted as Viral) for a topic
2. Gets the top 5 tweets by engagement for each event
3. Uses Claude API to extract specific, verifiable factual claims
4. Deduplicates claims across events
5. Outputs CSV + Interactive HTML timeline
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: uv add anthropic")
    sys.exit(1)

# Path setup
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# Load environment variables from .env file
load_dotenv(repo_root / ".env")


# =============================================================================
# CONFIGURATION
# =============================================================================
CLAUDE_MODEL = "claude-3-5-haiku-latest"
TOP_TWEETS_PER_EVENT = 5
PEEK_HOURS = 6
TOP_TOPICS = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]

# =============================================================================
# PROMPTS
# =============================================================================
CLAIM_EXTRACTION_PROMPT = """You are a fact-checker analyzing tweets from a viral social media moment.

Your task: Extract SPECIFIC FACTUAL CLAIMS from these tweets that could be fact-checked.

Requirements for a valid claim:
1. It must be a concrete, specific assertion (not a vague statement)
2. It must be verifiable as true or false
3. It must be directly stated or strongly implied in the tweet text
4. It should be a "load-bearing" claim - if proven false, the tweet's argument collapses

Good examples:
- "OpenAI's GPT-4 scored in the 90th percentile on the bar exam"
- "Google fired 12,000 employees in January 2023"
- "The AI-generated image won first place at the Colorado State Fair"

Bad examples (do NOT extract these):
- "AI is dangerous" (opinion, not verifiable)
- "This changes everything" (vague, not specific)
- "Many experts believe..." (weasel words, no specific claim)
- "AI technology is advancing" (too general)

TWEETS:
---
{tweets_formatted}
---

Extract 1-3 factual claims from the above tweets. For each claim:
1. State the claim clearly
2. Quote the specific tweet text that contains this claim

Format:
CLAIM: [the factual claim]
SOURCE: "[exact quote from tweet]"

If no specific factual claims can be extracted, respond with: NO_FACTUAL_CLAIMS"""


DEDUPLICATION_PROMPT = """You are organizing factual claims extracted from social media data about AI/technology topics.

Below are claims extracted from multiple viral events. Some claims may be:
- Duplicates (same claim, different wording)
- Related (different claims about the same topic/event)
- Unrelated (completely different topics)

Group the claims by topic/event and deduplicate similar ones.

CLAIMS:
{all_claims}

Output as JSON with this structure:
{{
  "groups": [
    {{
      "theme": "Brief description of the claim theme",
      "canonical_claim": "The clearest, most accurate version of the claim",
      "variants": ["list", "of", "original", "claim", "wordings"],
      "event_count": number_of_events_mentioning_this
    }}
  ]
}}

Return ONLY valid JSON, no other text."""


# =============================================================================
# DATA LOADING
# =============================================================================
def load_prediction_dataset() -> pl.DataFrame:
    """Load the prediction dataset from MLflow artifacts."""
    mlruns_dir = repo_root / "mlruns"

    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("."):
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir():
                    dataset_file = run_dir / "artifacts" / "prediction_dataset.csv"
                    if dataset_file.exists():
                        return pl.read_csv(dataset_file)

    raise FileNotFoundError("prediction_dataset.csv not found in any MLflow run")


def load_tweets_with_content() -> pl.DataFrame:
    """Load tweet assignments joined with raw tweet content."""
    # Load assignments
    assignments_path = repo_root / "data" / "tweet_topic_assignments_with_engagement.csv"
    assignments = pl.read_csv(assignments_path)

    # Parse timestamp
    assignments = assignments.with_columns(pl.col("timestamp").str.to_datetime().alias("timestamp"))

    # Load raw tweets for tweet text
    raw_path = repo_root / "data" / "raw" / "tweets_ai.parquet"
    raw_tweets = pl.read_parquet(raw_path)

    # Convert ID to int64 for join
    raw_tweets = raw_tweets.with_columns(pl.col("id").cast(pl.Int64).alias("tweet_id"))

    # Select needed columns from raw tweets
    raw_tweets = raw_tweets.select(["tweet_id", "tweet"])

    # Join
    tweets = assignments.join(raw_tweets, on="tweet_id", how="left")

    return tweets


def get_viral_anomalies(dataset: pl.DataFrame, topic: int) -> pl.DataFrame:
    """Get viral anomalies for a specific topic."""
    viral = dataset.filter((pl.col("topic") == topic) & (pl.col("target") == 1))
    return viral


def get_top_tweets_for_event(
    event_start: str,
    topic: int,
    tweets_df: pl.DataFrame,
    n: int = TOP_TWEETS_PER_EVENT,
) -> list[dict]:
    """Get top N tweets by engagement for an anomaly event."""
    # Parse event_start
    event_start_dt = pl.Series([event_start]).str.to_datetime()[0]
    event_end_dt = event_start_dt + timedelta(hours=PEEK_HOURS)

    # Filter tweets for this topic and time window
    event_tweets = tweets_df.filter(
        (pl.col("topic") == topic) & (pl.col("timestamp") >= event_start_dt) & (pl.col("timestamp") < event_end_dt)
    )

    # Sort by engagement and take top N
    event_tweets = event_tweets.sort("total_engagement", descending=True).head(n)

    return event_tweets.to_dicts()


# =============================================================================
# CLAUDE API FUNCTIONS
# =============================================================================
def create_client() -> anthropic.Anthropic:
    """Create Anthropic client, checking for API key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def extract_claims_from_tweets(
    tweets: list[dict],
    event_start: str,
    client: anthropic.Anthropic,
) -> list[dict]:
    """Extract factual claims from tweets using Claude API."""
    if not tweets:
        return []

    # Format tweets for prompt
    tweets_formatted = ""
    for i, tweet in enumerate(tweets, 1):
        tweet_text = tweet.get("tweet", "")
        engagement = tweet.get("total_engagement", 0)
        tweets_formatted += f"Tweet {i} (engagement: {engagement}):\n{tweet_text}\n\n"

    prompt = CLAIM_EXTRACTION_PROMPT.format(tweets_formatted=tweets_formatted)

    # Call Claude API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
            break
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("    ERROR: Rate limit exceeded, skipping event")
                return []
        except Exception as e:
            print(f"    ERROR: API call failed: {e}")
            return []

    # Parse response
    claims = parse_claim_response(response_text, tweets, event_start)
    return claims


def parse_claim_response(
    response_text: str,
    tweets: list[dict],
    event_start: str,
) -> list[dict]:
    """Parse Claude's response into structured claims."""
    if "NO_FACTUAL_CLAIMS" in response_text:
        return []

    claims = []

    # Parse CLAIM: ... SOURCE: ... format
    claim_pattern = r"CLAIM:\s*(.+?)(?=SOURCE:|CLAIM:|$)"
    source_pattern = r'SOURCE:\s*"([^"]+)"'

    claim_matches = re.findall(claim_pattern, response_text, re.DOTALL)
    source_matches = re.findall(source_pattern, response_text)

    for i, claim_text in enumerate(claim_matches):
        claim_text = claim_text.strip()
        if not claim_text:
            continue

        source_quote = source_matches[i] if i < len(source_matches) else ""

        # Find the tweet that contains this source quote
        tweet_id = None
        engagement = 0
        for tweet in tweets:
            if source_quote and source_quote.lower() in tweet.get("tweet", "").lower():
                tweet_id = tweet.get("tweet_id")
                engagement = tweet.get("total_engagement", 0)
                break

        # If no match found, use the first tweet as default
        if tweet_id is None and tweets:
            tweet_id = tweets[0].get("tweet_id")
            engagement = tweets[0].get("total_engagement", 0)

        claims.append(
            {
                "event_start": event_start,
                "claim": claim_text,
                "source_quote": source_quote,
                "tweet_id": tweet_id,
                "engagement": engagement,
            }
        )

    return claims


def deduplicate_claims(
    all_claims: list[dict],
    client: anthropic.Anthropic,
) -> list[dict]:
    """Deduplicate claims using Claude API."""
    if not all_claims:
        return []

    # Format claims for prompt
    claims_formatted = ""
    for i, claim in enumerate(all_claims, 1):
        claims_formatted += f"{i}. [{claim['event_start'][:10]}] {claim['claim']}\n"

    prompt = DEDUPLICATION_PROMPT.format(all_claims=claims_formatted)

    # Call Claude API
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text

        # Parse JSON response
        # Find JSON in response (in case there's extra text)
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("groups", [])
        else:
            print("WARNING: Could not parse deduplication response as JSON")
            return []

    except Exception as e:
        print(f"ERROR: Deduplication failed: {e}")
        return []


# =============================================================================
# OUTPUT GENERATION
# =============================================================================
def generate_csv(claims: list[dict], topic: int, output_path: Path) -> None:
    """Generate CSV output."""
    if not claims:
        print("  No claims to write to CSV")
        return

    rows = []
    for claim in claims:
        rows.append(
            {
                "topic": topic,
                "event_start": claim["event_start"],
                "claim": claim["claim"],
                "source_quote": claim["source_quote"],
                "tweet_id": claim["tweet_id"],
                "engagement": claim["engagement"],
            }
        )

    df = pl.DataFrame(rows)
    df.write_csv(output_path)
    print(f"  Saved CSV: {output_path}")


def generate_html_timeline(
    claims: list[dict],
    grouped_claims: list[dict],
    topic: int,
    output_path: Path,
) -> None:
    """Generate interactive HTML timeline."""

    # Group claims by event
    events_map = {}
    for claim in claims:
        event_start = claim["event_start"]
        if event_start not in events_map:
            events_map[event_start] = []
        events_map[event_start].append(claim)

    # Sort events by date
    sorted_events = sorted(events_map.keys())

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic {topic}: Viral Claims Timeline</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1a1a2e;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #1a1a2e;
            margin-bottom: 16px;
            font-size: 1.3em;
        }}
        .claim-group {{
            background: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 16px;
            margin-bottom: 12px;
            border-radius: 0 8px 8px 0;
        }}
        .claim-group h3 {{
            color: #2e7d32;
            font-size: 1em;
            margin-bottom: 8px;
        }}
        .claim-group .canonical {{
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }}
        .claim-group .meta {{
            font-size: 0.85em;
            color: #666;
        }}
        .timeline {{
            position: relative;
            padding-left: 30px;
        }}
        .timeline::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #ddd;
        }}
        .event {{
            position: relative;
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.2s;
        }}
        .event:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        .event::before {{
            content: '';
            position: absolute;
            left: -24px;
            top: 24px;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .event-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .event-date {{
            font-weight: 600;
            color: #1a1a2e;
        }}
        .event-badge {{
            background: #e8f5e9;
            color: #2e7d32;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }}
        .claims-list {{
            display: none;
        }}
        .claims-list.expanded {{
            display: block;
        }}
        .claim-item {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-top: 12px;
        }}
        .claim-text {{
            font-weight: 500;
            color: #333;
            margin-bottom: 8px;
        }}
        .claim-source {{
            font-style: italic;
            color: #666;
            font-size: 0.9em;
            padding: 8px;
            background: #fff;
            border-left: 3px solid #ccc;
            margin-top: 8px;
        }}
        .claim-meta {{
            font-size: 0.8em;
            color: #888;
            margin-top: 8px;
        }}
        .expand-hint {{
            color: #888;
            font-size: 0.85em;
            margin-top: 8px;
        }}
        .no-claims {{
            color: #888;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Topic {topic}: Factual Claims from Viral Anomalies</h1>
        <p class="subtitle">Extracted from {len(sorted_events)} viral events using Claude AI</p>

        <div class="summary">
            <h2>Deduplicated Claims Summary</h2>
"""

    # Add grouped claims summary
    if grouped_claims:
        for i, group in enumerate(grouped_claims, 1):
            theme = group.get("theme", "Unknown theme")
            canonical = group.get("canonical_claim", "")
            event_count = group.get("event_count", 0)

            html += f"""
            <div class="claim-group">
                <h3>Group {i}: {theme}</h3>
                <p class="canonical">{canonical}</p>
                <p class="meta">Appeared in {event_count} event(s)</p>
            </div>
"""
    else:
        html += '<p class="no-claims">No claim groups available</p>'

    html += """
        </div>

        <h2 style="margin-bottom: 20px;">Timeline</h2>
        <div class="timeline">
"""

    # Add events to timeline
    for event_start in sorted_events:
        event_claims = events_map[event_start]
        event_date = event_start[:10]
        claim_count = len(event_claims)

        html += f"""
            <div class="event" onclick="toggleClaims(this)">
                <div class="event-header">
                    <span class="event-date">{event_date}</span>
                    <span class="event-badge">{claim_count} claim(s)</span>
                </div>
                <p class="expand-hint">Click to expand claims</p>
                <div class="claims-list">
"""
        for claim in event_claims:
            claim_text = claim["claim"].replace('"', "&quot;")
            source_quote = claim["source_quote"].replace('"', "&quot;")
            engagement = claim["engagement"]

            html += f"""
                    <div class="claim-item">
                        <p class="claim-text">{claim_text}</p>
                        <div class="claim-source">"{source_quote}"</div>
                        <p class="claim-meta">Engagement: {engagement:,}</p>
                    </div>
"""
        html += """
                </div>
            </div>
"""

    html += """
        </div>
    </div>

    <script>
        function toggleClaims(element) {
            const claimsList = element.querySelector('.claims-list');
            const hint = element.querySelector('.expand-hint');
            if (claimsList.classList.contains('expanded')) {
                claimsList.classList.remove('expanded');
                hint.textContent = 'Click to expand claims';
            } else {
                claimsList.classList.add('expanded');
                hint.textContent = 'Click to collapse';
            }
        }
    </script>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"  Saved HTML: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main(topic: int) -> None:
    print("=" * 60)
    print(f"CLAIM EXTRACTION: Topic {topic}")
    print("=" * 60)

    # Create output directory
    output_dir = repo_root / "experiments" / "results" / "claims"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Claude client
    print("\n[1] Initializing Claude API client...")
    client = create_client()
    print("    Client initialized")

    # Load data
    print("\n[2] Loading data...")
    dataset = load_prediction_dataset()
    tweets_df = load_tweets_with_content()
    print(f"    Loaded {len(dataset)} prediction samples")
    print(f"    Loaded {len(tweets_df)} tweets")

    # Get viral anomalies for this topic
    print("\n[3] Getting viral anomalies...")
    viral_anomalies = get_viral_anomalies(dataset, topic)
    print(f"    Found {len(viral_anomalies)} viral anomalies for topic {topic}")

    if len(viral_anomalies) == 0:
        print("    No viral anomalies found. Exiting.")
        return

    # Extract claims for each event
    print("\n[4] Extracting claims from each event...")
    all_claims = []
    events = viral_anomalies.to_dicts()

    for i, event in enumerate(events):
        event_start = event["event_start"]
        print(f"    Event {i + 1}/{len(events)}: {event_start[:10]}...", end=" ")

        # Get top tweets
        top_tweets = get_top_tweets_for_event(event_start, topic, tweets_df)

        if not top_tweets:
            print("no tweets found")
            continue

        # Extract claims
        claims = extract_claims_from_tweets(top_tweets, event_start, client)
        print(f"{len(claims)} claims extracted")

        all_claims.extend(claims)

        # Small delay to avoid rate limits
        time.sleep(0.5)

    print(f"\n    Total claims extracted: {len(all_claims)}")

    if not all_claims:
        print("    No claims extracted. Exiting.")
        return

    # Deduplicate claims
    print("\n[5] Deduplicating claims...")
    grouped_claims = deduplicate_claims(all_claims, client)
    print(f"    Found {len(grouped_claims)} claim groups")

    # Generate outputs
    print("\n[6] Generating outputs...")
    csv_path = output_dir / f"claims_topic_{topic}.csv"
    html_path = output_dir / f"claims_topic_{topic}.html"

    generate_csv(all_claims, topic, csv_path)
    generate_html_timeline(all_claims, grouped_claims, topic, html_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Viral events processed: {len(events)}")
    print(f"Total claims extracted: {len(all_claims)}")
    print(f"Unique claim groups: {len(grouped_claims)}")
    print("\nOutputs:")
    print(f"  CSV:  {csv_path}")
    print(f"  HTML: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract factual claims from viral anomalies")
    parser.add_argument(
        "--topic", "-t", type=int, default=0, help=f"Topic ID to process (default: 0). Available: {TOP_TOPICS}"
    )
    parser.add_argument("--all", "-a", action="store_true", help="Process all top topics")

    args = parser.parse_args()

    if args.all:
        for topic in TOP_TOPICS:
            print(f"\n{'#' * 70}")
            print(f"# PROCESSING TOPIC {topic}")
            print(f"{'#' * 70}")
            main(topic)
    else:
        main(args.topic)
