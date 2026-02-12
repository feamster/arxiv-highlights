# arxiv-highlights

Automated weekly digest of papers at the intersection of **machine learning** and **computer networking** — curated from arXiv for [netml.io](https://netml.io).

## Latest Papers

Browse the [weekly/](./weekly/) directory for the latest digests, or check the [master index](./weekly/index.json).

Each week includes:
- **papers.json** — Machine-readable metadata, abstracts, and links
- **README.md** — Human-readable digest with author information

## How it works

1. Every Monday, a GitHub Action queries arXiv for recent papers matching NetML-relevant search terms
2. Results are deduplicated and filtered by date
3. Output is committed to this repo as a weekly digest
4. Optionally, Claude can generate concise summaries with inferred author affiliations

## Running locally

```bash
pip install -r requirements.txt
python fetch_papers.py
```

### CLI options

```bash
# Fetch last 7 days (default)
python fetch_papers.py

# Specific date range
python fetch_papers.py --start 2026-02-01 --end 2026-02-12

# Override max papers
python fetch_papers.py --max-papers 50

# Enable Claude summaries (requires ANTHROPIC_API_KEY env var)
python fetch_papers.py --summarize

# Dry run (fetch but don't write files)
python fetch_papers.py --dry-run
```

## Configuration

Edit `config.yaml` to adjust search queries, topic tags, or fetch settings.

## Setup for GitHub Actions

1. Go to repo Settings → Secrets → Actions
2. (Optional) Add `ANTHROPIC_API_KEY` as a repository secret for Claude summaries
3. The workflow runs automatically every Monday, or trigger manually from the Actions tab
