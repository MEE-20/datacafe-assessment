# F1 Capstone Template

Starter repository for the Cadra F1 walking-skeleton capstone. Use OpenCode with the Cadra provider to build your solution, document your approach, and submit a public GitHub repo for evaluation.

## Prerequisites

- Python 3.11+
- [OpenCode](https://opencode.ai) ≥ 1.17.0
- A Cadra JWT (`CADRA_TOKEN`) from the F1 Setup page
- Your Cadra proxy URL (from the F1 Setup page)

## Setup

1. Clone this repo (or use it as a GitHub template).
2. Set the environment variables (both values come from the F1 Setup page):
   ```bash
   export CADRA_PROXY_URL=<your-proxy-url>   # e.g. https://your-proxy.example.com/v1
   export CADRA_TOKEN=<your-cadra-jwt>
   ```
   `opencode.json` reads both via `{env:…}` — no file edits needed.
3. Install OpenCode if not already installed (see [opencode.ai](https://opencode.ai)).
4. Run OpenCode in this directory:
   ```bash
   opencode
   ```
5. Complete `APPROACH.md` and implement your solution in `src/` (start with `src/solution.py`).
6. Push your work to a **public** GitHub repository.
7. Submit your repo URL on the F1 demo page.

## Python environment (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

```
.
├── opencode.json    # Cadra provider config (reads CADRA_PROXY_URL + CADRA_TOKEN from env)
├── APPROACH.md      # Your written approach (required for submission)
├── src/
│   └── solution.py  # Your solution code
├── requirements.txt
└── README.md
```

## Live Endpoints (valid till 18 sept 2026 5pm)
the assistant is deployed on AWS EC2 instance(free tier)
http://65.2.35.190:8080/docs -- fastAPI swagger UI endpoint. It is accessible publically.
https://drives-suffered-batman-headquarters.trycloudflare.com - https - redirectec through cloudflare.
use: https://drives-suffered-batman-headquarters.trycloudflare.com/docs for proper UI testing
