#!/usr/bin/env python3
"""
fetch_papers.py — Fetch ML + Networking papers from arXiv

Queries arXiv API for recent papers matching NetML-relevant search terms,
optionally generates summaries with Claude, and outputs weekly digests.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import yaml

# Optional: Claude summarization
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def get_week_string(date: datetime) -> str:
    """Get ISO week string (e.g., '2026-W07') for a date."""
    return date.strftime("%Y-W%W")


def fetch_papers(
    queries: list[str],
    start_date: datetime,
    end_date: datetime,
    max_per_query: int = 50,
    max_total: int = 100,
    delay_seconds: float = 3.0,
    retries: int = 3,
) -> list[arxiv.Result]:
    """
    Fetch papers from arXiv matching the given queries.

    Runs each query separately, deduplicates by arXiv ID, and filters by date range.
    """
    seen_ids = set()
    all_papers = []
    client = arxiv.Client()

    for i, query in enumerate(queries):
        print(f"  Query {i + 1}/{len(queries)}: {query[:60]}...")

        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        query_count = 0
        for attempt in range(retries):
            try:
                for result in client.results(search):
                    # Filter by date range
                    pub_date = result.published.replace(tzinfo=timezone.utc)
                    if pub_date < start_date or pub_date > end_date:
                        continue

                    # Deduplicate by arXiv ID
                    paper_id = result.get_short_id()
                    if paper_id in seen_ids:
                        continue

                    seen_ids.add(paper_id)
                    all_papers.append(result)
                    query_count += 1

                    # Check total limit
                    if len(all_papers) >= max_total:
                        print(f"    Reached max total ({max_total}), stopping.")
                        return all_papers

                break  # Success, exit retry loop

            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay_seconds * 2)

        print(f"    Found {query_count} new papers")

        # Rate limiting between queries
        if i < len(queries) - 1:
            time.sleep(delay_seconds)

    return all_papers


def format_authors(result: arxiv.Result) -> str:
    """Format authors with affiliations when available."""
    author_strs = []
    for author in result.authors:
        name = author.name
        # Check for affiliations (often empty in arXiv data)
        if hasattr(author, 'affiliations') and author.affiliations:
            affiliations = ", ".join(author.affiliations)
            author_strs.append(f"{name} ({affiliations})")
        else:
            author_strs.append(name)
    return ", ".join(author_strs)


def summarize_with_claude(
    paper: arxiv.Result,
    config: dict,
    client: "anthropic.Anthropic",
) -> dict:
    """
    Use Claude to generate a summary and infer author affiliations.

    Returns dict with 'authors_with_affiliations', 'summary', and 'tags'.
    """
    tags_list = config.get("tags", [])

    # Build author string for prompt
    author_names = [a.name for a in paper.authors]
    author_string = ", ".join(author_names)

    # Get categories
    categories = [paper.primary_category] + [c for c in paper.categories if c != paper.primary_category]

    prompt = f"""You are summarizing a research paper for "NetML Weekly" — a curated digest
of papers at the intersection of machine learning and computer networking.

Given the paper metadata below, return ONLY valid JSON with these keys:

1. "authors_with_affiliations": Formatted author list string.
   If affiliations are in the metadata, use them.
   If not, infer likely affiliations from author names and paper context
   (many networking/ML researchers are well-known).
   If you can't confidently infer, just list names without affiliations.
   Format: "Author Name (University), Author Name (Company), ..."

2. "summary": 2-3 sentences for a technical audience (networking/ML researchers).
   Sentence 1: What the paper proposes or does.
   Sentence 2: The key method or approach.
   Sentence 3: Main results or findings. Be specific with numbers from the abstract.

3. "tags": 2-4 tags from this list: {tags_list}

---
Title: {paper.title}
Authors: {author_string}
Categories: {categories}
Abstract: {paper.summary}"""

    claude_config = config.get("claude", {})
    model = claude_config.get("model", "claude-sonnet-4-20250514")
    max_tokens = claude_config.get("max_tokens", 600)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        # Strip markdown code blocks if present
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        return json.loads(content)

    except Exception as e:
        print(f"    Claude summarization failed: {e}")
        # Fallback: use raw data
        return {
            "authors_with_affiliations": format_authors(paper),
            "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
            "tags": [],
        }


def cluster_papers(paper_dicts: list[dict], client: "anthropic.Anthropic") -> dict:
    """Use Claude to cluster papers into thematic sessions."""
    # Build a compact list of papers for Claude
    paper_list = "\n".join(
        f"{i+1}. {p['title']}"
        for i, p in enumerate(paper_dicts)
    )

    prompt = f"""You are organizing papers for "NetML Weekly" — a digest of ML + networking papers.

Group these {len(paper_dicts)} papers into 5-10 thematic sessions, like a conference program.
Each session should have 2-6 papers. Every paper must be assigned to exactly one session.

Return ONLY valid JSON with this structure:
{{
  "sessions": [
    {{
      "title": "Session Title (e.g., 'Traffic Analysis & Classification')",
      "description": "One sentence describing the session theme",
      "papers": [1, 5, 12]  // paper numbers from the list below
    }}
  ]
}}

Papers:
{paper_list}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'^```\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        return json.loads(content)

    except Exception as e:
        print(f"  Clustering failed: {e}")
        return None


def cluster_by_category(paper_dicts: list[dict]) -> dict:
    """Fallback: cluster papers by arXiv primary category."""
    category_names = {
        "cs.NI": "Networking",
        "cs.LG": "Machine Learning",
        "cs.CR": "Security & Privacy",
        "cs.AI": "Artificial Intelligence",
        "cs.DC": "Distributed Computing",
        "cs.IT": "Information Theory",
        "eess.SP": "Signal Processing",
        "stat.ML": "Statistical ML",
    }

    clusters = {}
    for i, p in enumerate(paper_dicts):
        cat = p["primary_category"]
        name = category_names.get(cat, cat)
        if name not in clusters:
            clusters[name] = []
        clusters[name].append(i + 1)

    return {
        "sessions": [
            {"title": name, "description": f"Papers in {name}", "papers": indices}
            for name, indices in clusters.items()
            if indices
        ]
    }


def paper_to_dict(paper: arxiv.Result, summary_data: dict = None) -> dict:
    """Convert an arxiv.Result to our JSON format."""
    # Use summary data if provided, otherwise use raw metadata
    if summary_data:
        authors = summary_data.get("authors_with_affiliations", format_authors(paper))
        summary = summary_data.get("summary", paper.summary)
        tags = summary_data.get("tags", [])
    else:
        authors = format_authors(paper)
        summary = paper.summary
        tags = []

    return {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": authors,
        "published": paper.published.strftime("%Y-%m-%d"),
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "arxiv_url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "abstract": paper.summary,
        "summary": summary,
        "tags": tags,
    }


def generate_readme(week_data: dict) -> str:
    """Generate the human-readable README.md for a week."""
    date_range = week_data["date_range"]
    papers = week_data["papers"]
    sessions = week_data.get("sessions")

    # Parse the start date for display
    start = datetime.strptime(date_range["start"], "%Y-%m-%d")
    header_date = start.strftime("%B %d, %Y")

    lines = [
        f"# NetML Weekly — Week of {header_date}",
        "",
        f"*{len(papers)} papers at the intersection of machine learning and computer networking.*",
        "",
    ]

    # Table of contents if we have sessions
    if sessions:
        lines.append("## Sessions")
        lines.append("")
        for session in sessions:
            anchor = session["title"].lower().replace(" ", "-").replace("&", "").replace("--", "-")
            lines.append(f"- [{session['title']}](#{anchor}) ({len(session['papers'])} papers)")
        lines.append("")

    lines.append("---")
    lines.append("")

    if sessions:
        # Organized by session
        for session in sessions:
            lines.append(f"## {session['title']}")
            lines.append("")
            if session.get("description"):
                lines.append(f"*{session['description']}*")
                lines.append("")

            for paper_num in session["papers"]:
                paper = papers[paper_num - 1]  # 1-indexed
                lines.append(f"### {paper['title']}")
                lines.append("")
                lines.append(f"**Authors:** {paper['authors']}")
                lines.append(f"[arXiv]({paper['arxiv_url']}) · [PDF]({paper['pdf_url']})")
                lines.append("")
                lines.append(paper["summary"])
                lines.append("")

            lines.append("---")
            lines.append("")
    else:
        # Flat list fallback
        for i, paper in enumerate(papers, 1):
            lines.append(f"### {i}. {paper['title']}")
            lines.append("")
            lines.append(f"**Authors:** {paper['authors']}")
            categories_str = ", ".join(paper["categories"])
            lines.append(f"**Published:** {paper['published']} | **Categories:** {categories_str}")
            lines.append(f"[arXiv]({paper['arxiv_url']}) · [PDF]({paper['pdf_url']})")
            lines.append("")
            lines.append(paper["summary"])
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def update_index(weekly_dir: Path, week: str, date_range: dict, paper_count: int):
    """Update the master index.json with a new week's entry."""
    index_path = weekly_dir / "index.json"

    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
    else:
        index = {"weeks": []}

    # Remove existing entry for this week if present
    index["weeks"] = [w for w in index["weeks"] if w["week"] != week]

    # Add new entry at the beginning
    index["weeks"].insert(0, {
        "week": week,
        "date_range": date_range,
        "paper_count": paper_count,
        "path": f"{week}/papers.json",
    })

    # Sort by week descending
    index["weeks"].sort(key=lambda w: w["week"], reverse=True)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ML + Networking papers from arXiv"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 7 days ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Max total papers to fetch.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Use Claude to generate summaries (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="Skip organizing papers into sessions (flat list instead).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process but don't write files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file.",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    fetch_config = config.get("fetch", {})

    # Determine date range
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59)

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        lookback = fetch_config.get("lookback_days", 7)
        start_date = (end_date - timedelta(days=lookback)).replace(hour=0, minute=0, second=0)

    # Determine max papers
    max_total = args.max_papers or fetch_config.get("max_total", 100)
    max_per_query = fetch_config.get("max_per_query", 50)
    delay_seconds = fetch_config.get("arxiv_delay_seconds", 3.0)
    retries = fetch_config.get("arxiv_retries", 3)

    print(f"NetML Weekly Paper Fetch")
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Max papers: {max_total}")
    print(f"  Summarization: {'enabled' if args.summarize else 'disabled (use --summarize to enable)'}")
    print()

    # Fetch papers from queries
    print("Fetching papers from arXiv...")
    queries = config.get("queries", [])
    papers = fetch_papers(
        queries=queries,
        start_date=start_date,
        end_date=end_date,
        max_per_query=max_per_query,
        max_total=max_total,
        delay_seconds=delay_seconds,
        retries=retries,
    )

    print(f"\nTotal papers found: {len(papers)}")

    if not papers:
        print("No papers found. Exiting.")
        return

    # Initialize Claude client if summarization is enabled
    claude_client = None
    if args.summarize:
        if not ANTHROPIC_AVAILABLE:
            print("Warning: anthropic package not installed. Skipping summarization.")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            print("Warning: ANTHROPIC_API_KEY not set. Skipping summarization.")
        else:
            claude_client = anthropic.Anthropic()
            print("\nGenerating summaries with Claude...")

    # Process papers
    paper_dicts = []
    for i, paper in enumerate(papers):
        print(f"  Processing {i + 1}/{len(papers)}: {paper.title[:50]}...")

        summary_data = None
        if claude_client:
            summary_data = summarize_with_claude(paper, config, claude_client)
            time.sleep(0.5)  # Rate limiting for Claude API

        paper_dicts.append(paper_to_dict(paper, summary_data))

    # Sort by published date (newest first)
    paper_dicts.sort(key=lambda p: p["published"], reverse=True)

    # Cluster papers into sessions
    sessions = None
    if not args.no_organize:
        print("\nOrganizing papers into sessions...")
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic()
            clustering = cluster_papers(paper_dicts, client)
            if clustering:
                sessions = clustering.get("sessions")
                print(f"  Created {len(sessions)} sessions")
        if not sessions:
            print("  Using arXiv categories as fallback...")
            clustering = cluster_by_category(paper_dicts)
            sessions = clustering.get("sessions")
            print(f"  Created {len(sessions)} sessions")

    # Determine week string
    week = get_week_string(end_date)

    # Build week data
    week_data = {
        "week": week,
        "date_range": {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        },
        "generated": datetime.now(timezone.utc).isoformat(),
        "paper_count": len(paper_dicts),
        "papers": paper_dicts,
        "sessions": sessions,
    }

    if args.dry_run:
        print("\n[DRY RUN] Would write the following:")
        print(f"  weekly/{week}/papers.json ({len(paper_dicts)} papers)")
        print(f"  weekly/{week}/README.md")
        print(f"  weekly/index.json (updated)")
        return

    # Write output
    weekly_dir = Path("weekly")
    week_dir = weekly_dir / week
    week_dir.mkdir(parents=True, exist_ok=True)

    # Write papers.json
    papers_path = week_dir / "papers.json"
    with open(papers_path, "w") as f:
        json.dump(week_data, f, indent=2)
    print(f"\nWrote {papers_path}")

    # Write README.md
    readme_content = generate_readme(week_data)
    readme_path = week_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Wrote {readme_path}")

    # Update index
    update_index(weekly_dir, week, week_data["date_range"], len(paper_dicts))
    print(f"Updated {weekly_dir / 'index.json'}")

    print(f"\nDone! {len(paper_dicts)} papers for {week}")


if __name__ == "__main__":
    main()
