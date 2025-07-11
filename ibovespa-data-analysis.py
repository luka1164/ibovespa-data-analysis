#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
from datetime import datetime as dt
import os

# ========================
# SETTINGS
# ========================

# Custom list of periods in trading days (e.g., 5 = 1 week, 22 = 1 month, 252 = 1 year)
periods = [5, 22, 66, 252, 504, 1260, 2520]  # Add or remove periods as needed

# Filters
volume_threshold = 1_000_000   # Minimum average volume in the last 252 trading days
min_years_history = 5          # Minimum years of price history (used for initial filtering)

# File paths
tickers_file = 'acoesibovespa.xlsx'
prices_file = 'cotacoes_trab.csv'
volume_file = 'volumedf.csv'
output_sectors_file = 'screener_sectors.xlsx'
output_stocks_file = 'screener_stocks.xlsx'

# ========================
# LOAD TICKERS
# ========================

# Load tickers from Excel file
dftickers = pd.read_excel(tickers_file, skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa': 'Sector'})
dftickers['Código'] = dftickers['Código'] + '.SA'
tickers = dftickers['Código'].to_list()

# ========================
# DOWNLOAD OR LOAD PRICE DATA
# ========================

# If the CSVs don't exist, download from Yahoo Finance
if not os.path.exists(prices_file) or not os.path.exists(volume_file):
    date_index = pd.date_range(start="1900-01-01", end=pd.Timestamp.today(), freq="D")
    paneldf = pd.DataFrame(index=date_index)
    volumedf = pd.DataFrame(index=date_index)

    for t in tickers:
        try:
            df = yf.download(t, period="max")
            paneldf = paneldf.join(df["Close"])
            volumedf = volumedf.join(df["Volume"])
        except Exception as e:
            print(f"Error downloading {t}: {e}")

    paneldf = paneldf.dropna(how='all')
    volumedf = volumedf.dropna(how='all')

    paneldf.to_csv(prices_file)
    volumedf.to_csv(volume_file)
else:
    paneldf = pd.read_csv(prices_file, index_col=0)
    volumedf = pd.read_csv(volume_file, index_col=0)

# ========================
# FILTER STOCKS BY HISTORY AND VOLUME
# ========================

min_days_history = 252 * min_years_history
valid_cols = []

# Filter: minimum data history
for col in paneldf.columns:
    if paneldf[col].dropna().shape[0] > min_days_history:
        valid_cols.append(col)

# Filter: minimum average volume over last 252 trading days
valid_cols = [
    col for col in valid_cols
    if volumedf[col].tail(252).mean() > volume_threshold
]

# Apply filtering to panel
filtered_paneldf = paneldf.loc[:, valid_cols]
filtered_paneldf.columns = filtered_paneldf.columns.str.replace('.SA', '', regex=False)

# ========================
# CALCULATE RETURNS BY SECTOR
# ========================

# Reload tickers with sectors
dftickers_original = pd.read_excel(tickers_file, skiprows=3)
dftickers_original = dftickers_original.rename(columns={'Setor Econômico\nBovespa': 'Sector'})
dftickers_original = dftickers_original[['Código', 'Sector']]

# Add sector info to stock prices
carteiradf = filtered_paneldf.T
carteiradf = carteiradf.join(dftickers_original.set_index('Código'))
carteiradf = carteiradf.dropna(subset=['Sector'])

# Prepare result table
sectors_summary = pd.DataFrame(index=carteiradf['Sector'].unique())

# Group by sector and calculate return for each period
sectors_grouped = carteiradf.groupby('Sector')

for name, group in sectors_grouped:
    group = group.drop(columns='Sector')
    group = group.T.sum(axis=1).sort_index().dropna()
    group = group[group > 0]

    for n in periods:
        n = min(n, len(group))
        ret = (group.iloc[-1] - group.iloc[-n]) / group.iloc[-n] * 100
        sectors_summary.loc[name, n] = round(ret, 2)

sectors_summary.to_excel(output_sectors_file)

# ========================
# CALCULATE RETURNS FOR INDIVIDUAL STOCKS
# ========================

stocks_summary = pd.DataFrame()

# Transpose to have time series in rows
carteiradf1 = carteiradf.drop(columns='Sector').T
carteiradf1 = carteiradf1.sort_index().dropna(how='all')

for c in carteiradf1.columns:
    serie = carteiradf1[c].dropna()
    if len(serie) < 2:
        continue
    for n in periods + [len(serie)]:
        n = min(n, len(serie))
        ret = (serie.iloc[-1] - serie.iloc[-n]) / serie.iloc[-n] * 100
        stocks_summary.loc[c, n] = round(ret, 2)

# Add metadata (name, sector)
dftickers_final = pd.read_excel(tickers_file, skiprows=3)
dftickers_final = dftickers_final.rename(columns={'Setor Econômico\nBovespa': 'Sector'})
dftickers_final = dftickers_final[['Nome', 'Código', 'Sector']]

stocks_summary = stocks_summary.join(dftickers_final.set_index('Código'))
stocks_summary.to_excel(output_stocks_file)

# ========================
# DONE
# ========================

print("Files generated:")
print(f"- {output_sectors_file}")
print(f"- {output_stocks_file}")
