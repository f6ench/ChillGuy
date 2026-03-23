# Betfair AI Trading Bot

Automated Betfair Exchange trading bot with Claude as the intelligence layer. Replaces static rules-based trading (like Bet Angel) with AI-driven probability estimation and execution.

## Architecture

```
Layer 1: Data Ingestion     → Betfair Streaming API (odds, volume, money flow)
Layer 2: Claude Analysis    → Probability estimation, edge calculation, strategy selection
Layer 3: Execution Engine   → Order placement, hedging, greening up
Layer 4: Feedback Loop      → Trade logging, bias detection, model improvement
Self-Healing:               → Auto-recovery, strategy switching, signal weight tuning
```

## Strategies

- **Scalping** — Back/lay close together for spread profit
- **Dobbing** — Back pre-race, lay at half odds in-play
- **Back to Lay** (Springer Bot) — Back a shortening horse, lay at lower price
- **Lay to Back** (Drifter Bot) — Lay a drifting horse, back at higher price

## Self-Healing System

- **Level 1 (Uptime)** — Auto-retry with exponential backoff, heartbeat monitoring, structural failure detection
- **Level 2 (Performance)** — Rolling win rate tracking, auto strategy switching, over-trading protection, emergency paper mode
- **Level 3 (Intelligence)** — Signal source accuracy tracking, dynamic weight adjustment, bias detection

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your Betfair API credentials and Anthropic API key
# Place Betfair SSL certificates in ./certs/
```

## Usage

```bash
# Paper trading (safe, no real money)
betfair-trader run --paper

# Live trading
betfair-trader run --live

# View performance
betfair-trader status

# Analyze prediction bias
betfair-trader bias
```

## Core Formula

```
EV = p × payout - (1 - p) × loss
edge = our_p - market_p
Rule: if edge > 0 → trade. If not → skip.
```

Position sizing uses fractional Kelly criterion (0.25x) to manage risk.

## Requirements

- Python 3.10+
- Betfair Exchange account with API access
- SSL certificates for Betfair login
- Anthropic API key (for Claude analysis layer)
