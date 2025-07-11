# IBOVESPA Data Analysis

This project downloads historical stock prices and volume data for stocks in the Bovespa index using the `yfinance` package, filters them by availability and liquidity, and saves the resulting data as CSVs.

## Features

- Downloads daily closing prices and volume data.
- Filters assets with at least 5 years of data.
- Filters assets with a minimum average volume of 1,000,000.
- Saves clean data to CSV for further analysis.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
