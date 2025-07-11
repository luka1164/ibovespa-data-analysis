#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
from datetime import datetime as dt
import os

# ========================
# CONFIGURAÇÕES GERAIS
# ========================

anos_maximos = 20                # Quantos anos para gerar períodos de retorno
volume_threshold = 1_000_000     # Volume médio mínimo nos últimos 252 pregões
anos_minimos = 5                 # Histórico mínimo de anos para filtrar ações

arquivo_tickers = 'acoesibovespa.xlsx'
arquivo_precos = 'cotacoes_trab.csv'
arquivo_volumes = 'volumedf.csv'
arquivo_screener_setores = 'screener_setores.xlsx'
arquivo_screener_acoes = 'tabela1trab.xlsx'

# ========================
# GERA LISTA DE PERÍODOS
# ========================

# Exemplo: [252, 504, 756, ..., 5040] se anos_maximos = 20
periodos = [252 * n for n in range(1, anos_maximos + 1)]

# ========================
# CARREGA OU CRIA BASE DE DADOS
# ========================

# Carregando lista de tickers com o setor associado
dftickers = pd.read_excel(arquivo_tickers, skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa': 'Setor'})
dftickers['Código'] = dftickers['Código'] + '.SA'
tickers = dftickers['Código'].to_list()

# Verifica se já existem arquivos salvos, para não baixar tudo novamente
if not os.path.exists(arquivo_precos) or not os.path.exists(arquivo_volumes):
    date_index = pd.date_range(start="1900-01-01", end=pd.Timestamp.today(), freq="D")
    paneldf = pd.DataFrame(index=date_index)
    volumedf = pd.DataFrame(index=date_index)

    for t in tickers:
        try:
            df = yf.download(t, period="max")
            paneldf = paneldf.join(df["Close"])
            volumedf = volumedf.join(df["Volume"])
        except Exception as e:
            print(f"Erro ao baixar {t}: {e}")

    paneldf = paneldf.dropna(how='all')
    volumedf = volumedf.dropna(how='all')

    paneldf.to_csv(arquivo_precos)
    volumedf.to_csv(arquivo_volumes)
else:
    paneldf = pd.read_csv(arquivo_precos, index_col=0)
    volumedf = pd.read_csv(arquivo_volumes, index_col=0)

# ========================
# FILTRAGEM DE AÇÕES
# ========================

# Seleção por histórico mínimo de pregões (5 anos equivalem a 252 * 5 dias)
min_count = 252 * anos_minimos
valid_cols = []

for col in paneldf.columns:
    if paneldf[col].dropna().shape[0] > min_count:
        valid_cols.append(col)

# Filtro por volume médio mínimo
valid_cols = [
    col for col in valid_cols
    if volumedf[col].tail(252).mean() > volume_threshold
]

filtered_paneldf = paneldf.loc[:, valid_cols]
filtered_paneldf.columns = filtered_paneldf.columns.str.replace('.SA', '', regex=False)

# ========================
# AGRUPAMENTO POR SETOR E CÁLCULO DE RETORNOS
# ========================

dftickers_original = pd.read_excel(arquivo_tickers, skiprows=3)
dftickers_original = dftickers_original.rename(columns={'Setor Econômico\nBovespa': 'Setor'})
dftickers_original = dftickers_original[['Código', 'Setor']]

carteiradf = filtered_paneldf.T
carteiradf = carteiradf.join(dftickers_original.set_index('Código'))
carteiradf = carteiradf.dropna(subset=['Setor'])

carteiras_sum = pd.DataFrame(index=carteiradf['Setor'].unique())

carteiras = carteiradf.groupby('Setor')

for name, group in carteiras:
    group = group.drop(columns='Setor')
    group = group.T.sum(axis=1).sort_index().dropna()
    group = group[group > 0]

    for n in periodos:
        n = min(n, len(group))
        retorno = (group.iloc[-1] - group.iloc[-n]) / group.iloc[-n] * 100
        carteiras_sum.loc[name, n] = round(retorno, 2)

carteiras_sum.to_excel(arquivo_screener_setores)

# ========================
# CÁLCULO DE RETORNOS INDIVIDUAIS DAS AÇÕES
# ========================

acoes_sum = pd.DataFrame()
carteiradf1 = carteiradf.drop(columns='Setor').T
carteiradf1 = carteiradf1.sort_index().dropna(how='all')

for c in carteiradf1.columns:
    serie = carteiradf1[c].dropna()
    if len(serie) < 2:
        continue
    for n in periodos + [len(serie)]:
        n = min(n, len(serie))
        retorno = (serie.iloc[-1] - serie.iloc[-n]) / serie.iloc[-n] * 100
        acoes_sum.loc[c, n] = round(retorno, 2)

# Inclui nome e setor para facilitar visualização no Excel final
dftickers_final = pd.read_excel(arquivo_tickers, skiprows=3)
dftickers_final = dftickers_final.rename(columns={'Setor Econômico\nBovespa': 'Setor'})
dftickers_final = dftickers_final[['Nome', 'Código', 'Setor']]

acoes_sum = acoes_sum.join(dftickers_final.set_index('Código'))
acoes_sum.to_excel(arquivo_screener_acoes)

print("Arquivos gerados:")
print(arquivo_screener_setores)
print(arquivo_screener_acoes)
