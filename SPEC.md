# arxiv-highlights — Full Build Spec

## Overview

Build a GitHub repo called `arxiv-highlights` that automatically fetches ML + Networking papers from arXiv weekly, uses Claude to generate concise summaries with author affiliations, and outputs a browsable weekly digest. Think "curated conference proceedings" for the NetML community, published at netml.io.

No PDFs stored. Just metadata, summaries, and links to arXiv.

---

## Repo Structure

```
arxiv-highlights/
├── README.md
├── requirements.txt
├── config.yaml                  # Search queries, tags, settings
├── fetch_papers.py              # Main script: fetch + summarize + output
├── weekly/                      # Auto-generated output directory
│   ├── index.json               # Master index of all weeks
│   ├── 2026-W07/
│   │   ├── papers.json          # Machine-readable paper data
│   │   └── README.md            # Human-readable weekly digest
│   ├── 2026-W08/
│   │   ├── papers.json
│   │   └── README.md
│   └── ...
├── .github/
│   └── workflows/
│       └── weekly-fetch.yml     # GitHub Actions: runs every Monday
└── .gitignore
```

---

## config.yaml

Controls everything about what gets fetched and how it's categorized.

```yaml
# config.yaml

# arXiv search queries. Each query is run separately, results are deduplicated.
# These define the scope of "NetML" — tune as the field evolves.
queries:
  # ML/DL papers in the networking category
  - 'cat:cs.NI AND ("machine learning" OR "deep learning" OR "neural network")'

  # Traffic analysis & classification
  - '("network traffic" OR "traffic classification" OR "traffic analysis") AND ("machine learning" OR "deep learning" OR "transformer")'

  # Internet / network measurement + ML
  - '("internet measurement" OR "network measurement") AND ("machine learning" OR "deep learning")'

  # Congestion control / transport + ML
  - '("congestion control" OR "transport protocol") AND ("reinforcement learning" OR "deep learning")'

  # SDN / network management + ML
  - '("software defined networking" OR "network management" OR "network operations") AND ("machine learning" OR "deep learning")'

  # Network security + ML
  - '("intrusion detection" OR "anomaly detection" OR "network security") AND ("machine learning" OR "deep learning" OR "neural network")'

  # Generative models for networking
  - '("network traffic" OR "packet" OR "netflow") AND ("generative" OR "diffusion model" OR "GAN" OR "large language model")'

  # DNS / CDN / Web + ML
  - '("DNS" OR "CDN" OR "web performance") AND ("machine learning" OR "deep learning")'

  # IoT + ML networking
  - '("IoT" OR "internet of things") AND ("network" OR "traffic") AND ("machine learning" OR "federated learning")'

  # Wireless / cellular + ML
  - '("wireless network" OR "cellular network" OR "5G" OR "spectrum") AND ("deep learning" OR "reinforcement learning")'

# Allowed topic tags for categorization. Claude picks 2-4 per paper.
tags:
  - traffic-classification
  - anomaly-detection
  - congestion-control
  - network-measurement
  - network-security
  - generative-models
  - federated-learning
  - reinforcement-learning
  - SDN
  - IoT
  - DNS
  - CDN
  - web-performance
  - routing
  - wireless
  - LLM
  - diffusion-models
  - graph-neural-networks
  - time-series
  - transformer
  - network-management
  - QoS
  - spectrum
  - edge-computing

# Fetch settings
fetch:
  max_per_query: 50        # Max results per individual query
  max_total: 100           # Max total papers per week
  lookback_days: 7         # Default date range
  arxiv_delay_seconds: 3.0 # Rate limiting (be nice to arXiv)
  arxiv_retries: 3

# Claude settings for summarization
claude:
  model: "claude-sonnet-4-20250514"
  max_tokens: 600
```

---

## fetch_papers.py

This is the main script. It does three things:

1. **Fetch** — Queries arXiv API using the `arxiv` Python library. Runs each query from config.yaml, deduplicates by arXiv ID, filters by date range.

2. **Summarize** — For each paper, sends the title + authors + abstract to the Anthropic API. Claude returns:
   - `authors_with_affiliations`: Formatted author list. Uses affiliations from arXiv metadata when available. When not available (common — arXiv affiliations are optional and spotty), Claude infers from context or just lists names.
   - `summary`: 2-3 sentence summary for a technical audience. Sentence 1: what the paper proposes/does. Sentence 2: the key method or approach. Sentence 3: main results with specific numbers when available.
   - `tags`: 2-4 topic tags from the allowed list in config.yaml.

3. **Output** — Writes `papers.json` and `README.md` for the week, and updates the master `weekly/index.json`.

### CLI interface

```
# Fetch last 7 days (default)
python fetch_papers.py

# Specific date range
python fetch_papers.py --start 2026-02-01 --end 2026-02-12

# Override max papers
python fetch_papers.py --max-papers 50

# Skip Claude summaries (metadata only — useful for testing)
python fetch_papers.py --no-summaries

# Dry run (fetch + summarize but don't write files)
python fetch_papers.py --dry-run
```

### Key implementation details

**arXiv fetching:**
- Use the `arxiv` Python library (pip install arxiv)
- Run each query separately with `arxiv.Search(..., sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)`
- Deduplicate across queries using `result.get_short_id()` as the key
- Filter by date range using `result.published`
- Respect rate limits: 3 second delay between API calls (configurable)
- Extract author affiliations via `author.affiliations` attribute when present (often empty)

**Claude summarization prompt:**

```
You are summarizing a research paper for "NetML Weekly" — a curated digest
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

3. "tags": 2-4 tags from this list: [paste allowed tags from config]

---
Title: {title}
Authors: {author_string}
Categories: {categories}
Abstract: {abstract}
```

Parse Claude's JSON response. If it wraps in markdown code blocks, strip them. If parsing fails, fall back to raw abstract truncated to 300 chars and author names without affiliations.

**Output formats:**

`papers.json` per week:
```json
{
  "week": "2026-W07",
  "date_range": {
    "start": "2026-02-05",
    "end": "2026-02-12"
  },
  "generated": "2026-02-12T08:00:00",
  "paper_count": 23,
  "papers": [
    {
      "id": "2602.12345v1",
      "title": "Paper Title Here",
      "authors": "Author One (MIT), Author Two (Google)",
      "published": "2026-02-11",
      "categories": ["cs.NI", "cs.LG"],
      "primary_category": "cs.NI",
      "arxiv_url": "https://arxiv.org/abs/2602.12345v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12345v1",
      "abstract": "Full abstract text...",
      "summary": "Claude-generated 2-3 sentence summary...",
      "tags": ["traffic-classification", "LLM"]
    }
  ]
}
```

`README.md` per week (this is the human-readable digest):

```markdown
# NetML Weekly — Week of February 5, 2026

*23 papers at the intersection of machine learning and computer networking.*

---

### 1. Paper Title Here

**Authors:** Author One (MIT), Author Two (Google)
**Published:** 2026-02-11 | **Categories:** cs.NI, cs.LG
[arXiv](https://arxiv.org/abs/2602.12345v1) · [PDF](https://arxiv.org/pdf/2602.12345v1)

Claude-generated 2-3 sentence summary here. First sentence says what the paper does. Second describes the approach. Third gives results with numbers.

`traffic-classification` · `LLM`

---

### 2. Next Paper Title...
```

`weekly/index.json` (master index, updated each run):
```json
{
  "weeks": [
    {
      "week": "2026-W08",
      "date_range": {"start": "2026-02-12", "end": "2026-02-19"},
      "paper_count": 19,
      "path": "2026-W08/papers.json"
    },
    {
      "week": "2026-W07",
      "date_range": {"start": "2026-02-05", "end": "2026-02-12"},
      "paper_count": 23,
      "path": "2026-W07/papers.json"
    }
  ]
}
```

---

## .github/workflows/weekly-fetch.yml

Runs every Monday at 8am Central (14:00 UTC). Also supports manual trigger with optional date override.

```yaml
name: NetML Weekly Papers

on:
  schedule:
    - cron: '0 14 * * 1'  # Monday 8am CT
  workflow_dispatch:
    inputs:
      start_date:
        description: 'Start date (YYYY-MM-DD). Default: 7 days ago.'
        required: false
      end_date:
        description: 'End date (YYYY-MM-DD). Default: today.'
        required: false
      max_papers:
        description: 'Max papers to fetch.'
        required: false
        default: '100'

jobs:
  fetch-papers:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch and summarize papers
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          ARGS=""
          [ -n "${{ github.event.inputs.start_date }}" ] && ARGS="$ARGS --start ${{ github.event.inputs.start_date }}"
          [ -n "${{ github.event.inputs.end_date }}" ] && ARGS="$ARGS --end ${{ github.event.inputs.end_date }}"
          [ -n "${{ github.event.inputs.max_papers }}" ] && ARGS="$ARGS --max-papers ${{ github.event.inputs.max_papers }}"
          python fetch_papers.py $ARGS

      - name: Commit and push
        run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"
          git add weekly/
          WEEK=$(date +%Y-W%V)
          git diff --staged --quiet || git commit -m "📚 NetML Weekly — $WEEK"
          git push
```

---

## requirements.txt

```
arxiv>=2.1.0
anthropic>=0.40.0
pyyaml>=6.0
```

---

## README.md (repo root)

```markdown
# arxiv-highlights

Automated weekly digest of papers at the intersection of **machine learning** and **computer networking** — curated from arXiv for [netml.io](https://netml.io).

## 📚 Latest Papers

Browse the [weekly/](./weekly/) directory for the latest digests, or check the [master index](./weekly/index.json).

Each week includes:
- **papers.json** — Machine-readable metadata, summaries, and links
- **README.md** — Human-readable digest with author affiliations and concise summaries

## How it works

1. Every Monday, a GitHub Action queries arXiv for recent papers matching NetML-relevant search terms
2. Results are deduplicated and filtered by date
3. Claude generates a 2-3 sentence summary of each paper with inferred author affiliations
4. Output is committed to this repo as a weekly digest

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python fetch_papers.py
```

## Configuration

Edit `config.yaml` to adjust search queries, topic tags, or fetch settings.

## Setup for GitHub Actions

1. Go to repo Settings → Secrets → Actions
2. Add `ANTHROPIC_API_KEY` as a repository secret
3. The workflow runs automatically every Monday, or trigger manually from the Actions tab
```

---

## .gitignore

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
```

---

## Setup Instructions (for Claude Code)

1. Create the GitHub repo `netml/arxiv-highlights` (or whatever org/name you prefer)
2. Create all files listed above
3. The repo needs one secret configured: `ANTHROPIC_API_KEY` — add this in GitHub repo Settings → Secrets and variables → Actions
4. To test locally: `pip install -r requirements.txt && ANTHROPIC_API_KEY=sk-... python fetch_papers.py`
5. To trigger manually: Go to Actions tab → "NetML Weekly Papers" → "Run workflow"

That's everything. The first run will create the `weekly/` directory structure and the first week's digest.
