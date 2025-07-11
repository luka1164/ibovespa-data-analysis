#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime as dt, timedelta as td
from scipy.stats import norm
from scipy import optimize


# In[2]:


dftickers = pd.read_excel('acoesibovespa.xlsx', skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa':'Setor'})
dftickers1 = dftickers.copy()
dftickers1['Código'] = dftickers1['Código'] + '.SA'
tickers = dftickers1['Código'].to_list()


# In[106]:


date_index = pd.date_range(start="1900-01-01", end=pd.Timestamp.today(), freq="D")
paneldf = pd.DataFrame(index=date_index)

for t in tickers:
    try:
        df = yf.download(t, period="max")["Close"]
        paneldf = paneldf.join(df)
    except:
        pass

paneldf = paneldf.dropna(how='all')
paneldf.to_csv('cotacoes_trab.csv')


# In[35]:


date_index = pd.date_range(start="1900-01-01", end=pd.Timestamp.today(), freq="D")
volumedf = pd.DataFrame(index=date_index)

for t in tickers:
    try:
        df = yf.download(t, period="max")["Volume"]
        volumedf = volumedf.join(df)
    except:
        pass

volumedf = volumedf.dropna(how='all')
volumedf.to_csv('volumedf.csv')


# In[3]:


paneldf = pd.read_csv('cotacoes_trab.csv', index_col=0)
volumedf = pd.read_csv('volumedf.csv', index_col=0)
print(len(paneldf.columns))


# In[4]:


paneldf = pd.read_csv('cotacoes_trab.csv', index_col=0)
volumedf = pd.read_csv('volumedf.csv', index_col=0)

min_count = 252 * 5
valid_cols = []

for col in paneldf.columns:
    if len(paneldf[col].dropna()) > min_count:
        valid_cols.append(col)
print(len(valid_cols))
paneldf = paneldf.loc[:, valid_cols]

volume_threshold = 1000000
valid_cols = []

for col in paneldf.columns:
    if volumedf[col].tail(252).mean() > volume_threshold:
        valid_cols.append(col)
print(len(valid_cols))

filtered_paneldf= paneldf.loc[:, valid_cols]


# In[5]:


filtered_paneldf.columns = filtered_paneldf.columns.str.replace('.SA', '', regex=False)


# In[6]:


dftickers = pd.read_excel('acoesibovespa.xlsx', skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa':'Setor'})
dftickers = dftickers[['Código', 'Setor']]

carteiras_sum = pd.DataFrame(index=dftickers['Setor'].unique())
carteiradf = filtered_paneldf.T
carteiradf = carteiradf.join(dftickers.set_index('Código'))
carteiradf = carteiradf[:-1]
carteiras = carteiradf.groupby('Setor')


# In[7]:


for name, group in carteiras:
    group = group.drop(columns='Setor')
    group = group.T
    group = group.sum(axis=1)
    group = group.sort_index()
    group = group.dropna(how='all')
    group = group[group>0]

    for n in [252, 252*5, 252*10, 252*20]:

        if n > len(group):
            xx = (group.iloc[-1] - group.iloc[0]) / group.iloc[0]
        else:
            xx = (group.iloc[-1] - group.iloc[-n]) / group.iloc[-n]
        xx = xx*100
        xx = xx.round(2)

        carteiras_sum.loc[name,n] = xx

# carteiras_sum.sort_values(len(filtered_paneldf))
carteiras_sum.to_excel('screener_setores.xlsx')
carteiras_sum


# In[8]:


# Criando um DataFrame para o screener

acoes_sum = pd.DataFrame()

# Criando um DataFrame para o screener

carteiradf1 = carteiradf.drop(columns='Setor')
carteiradf1 = carteiradf1.T
carteiradf1 = carteiradf1.sort_index()
carteiradf1 = carteiradf1.dropna(how='all')

for c in carteiradf1.columns:
    acao_s = carteiradf1[c]
    acao_s = acao_s[acao_s>0]
    acao_s = acao_s.dropna()
    if len(acao_s) < 2:
        continue
    for n in [252, 252*5, 252*10, len(filtered_paneldf)]:
        if n > len(acao_s):
            xx = (acao_s.iloc[-1] - acao_s.iloc[0]) / acao_s.iloc[0]
        else:
            xx = (acao_s.iloc[-1] - acao_s.iloc[-n]) / acao_s.iloc[-n]
        xx = xx*100
        xx = xx.round(2)
        acoes_sum.loc[c,n] = xx

pd.set_option('display.float_format', '{:.2f}'.format)
acoes_sum.sort_values(len(paneldf))

dftickers = pd.read_excel('acoesibovespa.xlsx', skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa':'Setor'})
dftickers = dftickers[['Nome', 'Código', 'Setor']]

acoes_sum.join(dftickers.set_index('Código')).to_excel('tabela1trab.xlsx')


# In[9]:


risk_free_rate = 0.10

carteiradf1 = carteiradf.drop(columns='Setor')
carteiradf1 = carteiradf1.T
carteiradf1 = carteiradf1.sort_index()
carteiradf1 = carteiradf1.dropna(how='all')

sharpe_ratios = pd.DataFrame()

for c in carteiradf1.columns:
    acao_s = carteiradf1[c]
    acao_s = acao_s[acao_s > 0].dropna()

    if len(acao_s) < 2:
        continue

    daily_returns = acao_s.pct_change().dropna()

    for n in [252, 252*5, 252*10, len(filtered_paneldf)]:
        if len(daily_returns) < n:
            sample = daily_returns
        else:
            sample = daily_returns[-n:]

        mean_return_daily = sample.mean()
        std_return_daily = sample.std()

        # Sharpe Ratio anualizado
        if std_return_daily > 0:
            sharpe = ((mean_return_daily * 252) - risk_free_rate) / (std_return_daily * np.sqrt(252))
            sharpe = round(sharpe, 3)
            sharpe_ratios.loc[c, n] = sharpe

# Formatação e exportação
pd.set_option('display.float_format', '{:.3f}'.format)

# Importar informações dos tickers
dftickers = pd.read_excel('acoesibovespa.xlsx', skiprows=3)
dftickers = dftickers.rename(columns={'Setor Econômico\nBovespa':'Setor'})
dftickers = dftickers[['Nome', 'Código', 'Setor']]

# Combinar e exportar
sharpe_ratios = sharpe_ratios.join(dftickers.set_index('Código'))
sharpe_ratios.to_excel('sharpe_ratios_tabela.xlsx')


# In[10]:


sharpe_ratios.sort_values(252)


# In[11]:


corrdf = pd.DataFrame()

for name, group in carteiras:
    group = group.drop(columns='Setor')
    group = group.T
    group = group.sum(axis=1)
    corrdf[name] = group

corrdf = corrdf[corrdf.index>'2022-01-01'].corr()
corrdf.to_excel('corrdf.xlsx')
corrdf


# In[16]:


df = sharpe_ratios.copy()

df = df.sort_values(by=["Setor", 252], ascending=[True, False])

top3_list = [group.head(3) for _, group in df.groupby("Setor")]
top3_df = pd.concat(top3_list, axis=0).reset_index(drop=False)
top3_df = top3_df[top3_df[252]>0]
top3_df.to_excel('portfolio_selecionado.xlsx')
portfolio_list = top3_df['index'].to_list()


# In[17]:


portdf = filtered_paneldf[portfolio_list].tail(252)
ret = portdf.pct_change()
medias = ret.mean()
cov = ret.cov()
desvios_padrao = ret.std()


# In[18]:


np.random.seed(1)
pesos = np.random.random((len(portfolio_list),1))
pesos /= np.sum(pesos)

for ticker, w in zip(tickers,pesos):
    print(f'Peso {ticker}: {w[0]:.5f}') 

ret_medio_port = ret.mean().dot(pesos) 
print(f'\nRetorno esperado do portfólio : {ret_medio_port[0]*100:.4f}%')
vol_port = np.sqrt(pesos.T.dot(ret.cov().dot(pesos)))
print(f'\nVolatilidade do portfólio: {vol_port[0][0]*100:.4f}%')
sr = ret_medio_port / vol_port
print(f'\nÍndice de Sharpe do portfólio: {sr[0][0]*100:.4f}%')


# In[19]:


n = 50_000

pesos_port = np.zeros(shape=(n,len(portdf.columns)))
volatilidade_port = np.zeros(n)
retorno_medio_port = np.zeros(n)
sr_port = np.zeros(n)

num_ativos = len(portdf.columns) # número de ativos igual ao número de colunas do dataframe
for i in range(n):
    # Peso de cada ativo
    pesos = np.random.random(num_ativos)
    # Normalize os pesos, para que a soma seja um.
    pesos /= np.sum(pesos)
    pesos_port[i,:] = pesos 

    # Retorno percentual esperado (soma ponderada dos retornos médios).
    ret_medio_port = ret.mean().dot(pesos)
    retorno_medio_port[i] = ret_medio_port

    # Volatilidade do portfólio
    vol_port = np.sqrt(pesos.T.dot(ret.cov().dot(pesos)))
    volatilidade_port[i] = vol_port

    # Indice de Sharpe
    sr = ret_medio_port / vol_port
    sr_port[i] = sr


plt.figure(figsize=(20,15))
plt.scatter(volatilidade_port,retorno_medio_port,c=sr_port, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility', fontsize=15)
plt.ylabel('Expected Return', fontsize=15)
plt.title('Expected Return and Volatility', fontsize=15)
plt.savefig('expreturn_vol.png', dpi=300, bbox_inches='tight')


# In[20]:


from scipy import optimize

def get_ret_vol_sr(pesos):
    pesos = np.array(pesos)
    ret = medias.dot(pesos)
    vol = np.sqrt(pesos.T.dot(cov.dot(pesos)))
    sr = ret / vol
    return np.array([ret, vol, sr])

def neg_sr(pesos):
    return get_ret_vol_sr(pesos)[2] * -1 

def check_sum(pesos):
    return np.sum(pesos) - 1

cons = {'type':'eq','fun':check_sum} # impõe a restrição 'check_sum = 0', soma dos pesos = 1
bounds = tuple((0, 1) for _ in range(23)) # impõe a restrição de pesos entre 0 e 1
init_guess = [.25 for _ in range(len(portdf.columns))]

opt_results = optimize.minimize(neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')
fronteira_ret = np.linspace(retorno_medio_port.min(), retorno_medio_port.max(), 100)
fronteira_vol = []

def minimize_vol(pesos):
    return get_ret_vol_sr(pesos)[1]

for possivel_ret in fronteira_ret:
    cons = ({'type':'eq','fun':check_sum},
            {'type':'eq','fun':lambda w:get_ret_vol_sr(w)[0] - possivel_ret})
    result = optimize.minimize(minimize_vol, init_guess, method='SLSQP', constraints=cons, bounds=bounds)
    fronteira_vol.append(result['fun'])

plt.figure(figsize=(20,15))
plt.scatter(volatilidade_port,retorno_medio_port,c=sr_port, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility', fontsize=15)
plt.ylabel('Expected Return', fontsize=15)
plt.title('Efficient Frontier', fontsize=15)

plt.plot(fronteira_vol, fronteira_ret, c='red', ls='--', lw=3, label='Fronteira Eficiente de Ativos com Risco')
plt.legend()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')


# In[21]:


def var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1):

    alfa = norm.ppf(1 - nivel_confianca, loc=0, scale=1)
    if relativo == 1:
        var = valor_inicial * -alfa * desvio_padrao * (periodos ** 0.5) # 'VaR relativo' (default)
    elif relativo == 0:
        var = valor_inicial * (-alfa * desvio_padrao * (periodos ** 0.5) - media * periodos) # 'VaR absoluto ou "zero"'
    return var

pesos = np.full((23, 1), 1/23)
for ticker, w in zip(tickers,pesos):
    print(f'Peso {ticker}: {w[0]*100:.2f}%')

soma = np.sum(pesos)
print(f'Soma dos pesos: {soma*100:.2f}%')

ret_medio_port = ret.mean().dot(pesos)
print(f'\nRetorno esperado do portfólio : {ret_medio_port[0]*100:.4f}%')

vol_port = np.sqrt(pesos.T.dot(ret.cov().dot(pesos)))
print(f'\nVolatilidade do portfólio: {vol_port[0][0]*100:.4f}%')

sr = ret_medio_port / vol_port
print(f'\nÍndice de Sharpe do portfólio: {sr[0][0]:.4f}')

valor_inicial = 100000
nivel_confianca = 0.95
desvio_padrao = vol_port[0][0]
media = ret_medio_port[0]
periodos = 1
var_relativo = var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
print(f'\nValor inicial: {valor_inicial}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo:.2f}')


# In[24]:


def get_ret_vol_sr(pesos):
    pesos = np.array(pesos)
    ret = medias.dot(pesos)
    vol = np.sqrt(pesos.T.dot(cov.dot(pesos)))
    sr = ret / vol
    return np.array([ret, vol, sr])

# seleciona o 3o. elemento (posição '2') do array acima = 'sr' e multiplica por -1 (negativo da função)
def neg_sr(pesos):
    return get_ret_vol_sr(pesos)[2] * -1 

def check_sum(pesos):
    return np.sum(pesos) - 1

cons = {'type':'eq','fun':check_sum} # impõe a restrição 'check_sum = 0', soma dos pesos = 1
bounds = tuple((0, 1) for _ in range(23)) # impõe a restrição de pesos entre 0 e 1
init_guess = [.25 for _ in range(23)]

opt_results = optimize.minimize(neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')

# Este bloco de código calcula os valores de volatilidade ao longo da "Fronteira Eficiente de Ativos com Risco",
# variando os retornos esperados. Ele utiliza o método de mínimos quadrados sequenciais (SLSQP) do SciPy 
# para minimizar a volatilidade sujeita à restrição de retorno desejado.
# Isso nos permite encontrar as carteiras de mínima volatilidade para cada retorno esperado possível.

fronteira_ret = np.linspace(retorno_medio_port.min(), retorno_medio_port.max(), 100)
fronteira_vol = []

# seleciona o 2o. elemento (posição '1') do array ([ret, vol, sr]))
def minimize_vol(pesos):
    return get_ret_vol_sr(pesos)[1]

for possivel_ret in fronteira_ret:
    cons = ({'type':'eq','fun':check_sum},
            {'type':'eq','fun':lambda w:get_ret_vol_sr(w)[0] - possivel_ret})
    result = optimize.minimize(minimize_vol, init_guess, method='SLSQP', constraints=cons, bounds=bounds)
    fronteira_vol.append(result['fun'])

plt.figure(figsize=(20,15))
plt.scatter(volatilidade_port,retorno_medio_port,c=sr_port, cmap='plasma')
plt.colorbar(label='"Indice de Sharpe" (excluindo a "taxa livre de risco")')
plt.xlabel('Volatilidade', fontsize=15)
plt.ylabel('Retorno esperado', fontsize=15)
plt.title('Fronteira Eficiente de Ativos com Risco', fontsize=15)

plt.plot(fronteira_vol, fronteira_ret, c='magenta', ls='--', lw=3, label='Fronteira Eficiente de Ativos com Risco')
plt.legend()


# In[ ]:


def var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1):

    alfa = norm.ppf(1 - nivel_confianca, loc=0, scale=1)

    # O VaR delta-normal com retornos i.i.d pode ser calculado de duas formas
    if relativo == 1:
        var = valor_inicial * -alfa * desvio_padrao * (periodos ** 0.5) # 'VaR relativo' (default)
    elif relativo == 0:
        var = valor_inicial * (-alfa * desvio_padrao * (periodos ** 0.5) - media * periodos) # 'VaR absoluto ou "zero"'

    return var

# Escolhendo o peso a cada ativo:
pesos = np.array([[0.125], [0.125], [0.125], [0.125], [0.125], [0.125], [0.125], [0.125]])
for ticker, w in zip(tickers,pesos):
    print(f'Peso {ticker}: {w[0]*100:.2f}%')

print()
# soma dos pesos
soma = np.sum(pesos)
print(f'Soma dos pesos: {soma*100:.2f}%')

# Retorno percentual esperado (soma ponderada dos retornos médios).
ret_medio_port = ret.mean().dot(pesos) 
print(f'\nRetorno esperado do portfólio : {ret_medio_port[0]*100:.4f}%')

# Volatilidade do portfólio
vol_port = np.sqrt(pesos.T.dot(ret.cov().dot(pesos)))
print(f'\nVolatilidade do portfólio: {vol_port[0][0]*100:.4f}%')

# Índice de Sharpe (Sharpe Ratio - sr)
sr = ret_medio_port / vol_port
print(f'\nÍndice de Sharpe do portfólio: {sr[0][0]:.4f}')

# O Valor em Risco (VaR) delta normal i.i.d. do portfólio
valor_inicial = 1000
nivel_confianca = 0.95
desvio_padrao = vol_port[0][0]
media = ret_medio_port[0]
periodos = 1
var_relativo = var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
print(f'\nValor inicial: {valor_inicial}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo:.2f}')


# In[26]:


def get_ret_vol_sr(pesos):
    pesos = np.array(pesos)
    ret = medias.dot(pesos)
    vol = np.sqrt(pesos.T.dot(cov.dot(pesos)))
    sr = ret / vol
    return np.array([ret, vol, sr])

# seleciona o 3o. elemento (posição '2') do array acima = 'sr') e multiplica por -1 (negativo da função)
def neg_sr(pesos):
    return get_ret_vol_sr(pesos)[2] * -1

def check_sum(pesos):
    return np.sum(pesos) - 1

def vol_fixa(pesos):
    return get_ret_vol_sr(pesos)[1] - vol_port[0][0]

cons = ({'type':'eq','fun':check_sum},
        {'type':'eq','fun':vol_fixa})
bounds = tuple((0, 1) for _ in range(23))
init_guess = [.25 for _ in range(23)]

opt_results = optimize.minimize(neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')

# Escolhendo o peso a cada ativo:
pesos_otimos_exemplo = opt_results.x
pesos_otimos_exemplo = pesos_otimos_exemplo.reshape(-1, 1)
for ticker, w in zip(tickers,pesos_otimos_exemplo):
    print(f'Peso {ticker}: {w[0]*100:.2f}%') 

print()
# soma dos pesos
soma = np.sum(pesos_otimos_exemplo)
print(f'Soma dos pesos: {soma*100:.2f}%')

# Retorno percentual esperado (soma ponderada dos retornos médios).
ret_medio_port_exemplo = ret.mean().dot(pesos_otimos_exemplo) 
print(f'\nRetorno esperado do portfólio : {ret_medio_port_exemplo[0]*100:.4f}%')

# Volatilidade do portfólio
vol_port_exemplo = np.sqrt(pesos_otimos_exemplo.T.dot(ret.cov().dot(pesos_otimos_exemplo)))
print(f'\nVolatilidade do portfólio: {vol_port_exemplo[0][0]*100:.4f}%')

# Índice de Sharpe (Sharpe Ratio - sr)
sr_exemplo = ret_medio_port_exemplo / vol_port_exemplo
print(f'\nÍndice de Sharpe do portfólio: {sr_exemplo[0][0]:.4f}')

# O Valor em Risco (VaR) delta normal i.i.d. do portfólio
valor_inicial = 100000 # valor atual do investimento na carteira em $
nivel_confianca = 0.95
desvio_padrao = vol_port_exemplo[0][0]
media = ret_medio_port_exemplo[0]
periodos = 1 # T
var_relativo = var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
print(f'\nValor inicial: {valor_inicial}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo:.2f}')

# O Valor em Risco (VaR) delta normal i.i.d. dos ativos individuais e sua soma
# utiliza os mesmos parâmetros acima de valor inicial, nível de confiança e núm. de períodos

var_relativo_individuais = []

for ticker, w, med, desv in zip(tickers, pesos_otimos_exemplo, medias, desvios_padrao):
    valor_inicial_acao = valor_inicial * w[0] # Valor inicial = valor atual do investimento na carteira em $, definido acima
    desvio_padrao = desv
    media = med
    var_relativo_ind = var_delta_normal_iid(valor_inicial_acao, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
    print(f'{ticker} - Valor inicial: {valor_inicial_acao:.2f}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo_ind:.2f}')
    var_relativo_individuais.append(var_relativo_ind)

print()
# Calcula a soma dos valore de VaR individuais
soma_var_relativo_individuais = sum(var_relativo_individuais)
print(f"Soma dos valores de VaR relativo individuais: {soma_var_relativo_individuais:.2f}")

print(f'\nEfeito de diversificação: {soma_var_relativo_individuais - var_relativo:.2f}')


# In[ ]:





# In[31]:


cons = ({'type':'eq','fun':check_sum},
        {'type':'eq','fun':vol_fixa})
bounds = tuple((0, 1) for _ in range(23))
init_guess = [.25 for _ in range(23)]

opt_results = optimize.minimize(neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')

pesos_otimos_exemplo = opt_results.x
pesos_otimos_exemplo = pesos_otimos_exemplo.reshape(-1, 1)
for ticker, w in zip(tickers,pesos_otimos_exemplo):
    print(f'Peso {ticker}: {w[0]*100:.2f}%') 

print()
# soma dos pesos
soma = np.sum(pesos_otimos_exemplo)
print(f'Soma dos pesos: {soma*100:.2f}%')

# Retorno percentual esperado (soma ponderada dos retornos médios).
ret_medio_port_exemplo = ret.mean().dot(pesos_otimos_exemplo) 
print(f'\nRetorno esperado do portfólio : {ret_medio_port_exemplo[0]*100:.4f}%')

# Volatilidade do portfólio
vol_port_exemplo = np.sqrt(pesos_otimos_exemplo.T.dot(ret.cov().dot(pesos_otimos_exemplo)))
print(f'\nVolatilidade do portfólio: {vol_port_exemplo[0][0]*100:.4f}%')

# Índice de Sharpe (Sharpe Ratio - sr)
sr_exemplo = ret_medio_port_exemplo / vol_port_exemplo
print(f'\nÍndice de Sharpe do portfólio: {sr_exemplo[0][0]:.4f}')

# O Valor em Risco (VaR) delta normal i.i.d. do portfólio
valor_inicial = 1000 # valor atual do investimento na carteira em $
nivel_confianca = 0.95
desvio_padrao = vol_port_exemplo[0][0]
media = ret_medio_port_exemplo[0]
periodos = 1 # T
var_relativo = var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
print(f'\nValor inicial: {valor_inicial}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo:.2f}')

var_relativo_individuais = []

for ticker, w, med, desv in zip(tickers, pesos_otimos_exemplo, medias, desvios_padrao):
    valor_inicial_acao = valor_inicial * w[0] # Valor inicial = valor atual do investimento na carteira em $, definido acima
    desvio_padrao = desv
    media = med
    var_relativo_ind = var_delta_normal_iid(valor_inicial_acao, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
    print(f'{ticker} - Valor inicial: {valor_inicial_acao:.2f}, VaR Rel. (DP amost.) - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo_ind:.2f}')
    var_relativo_individuais.append(var_relativo_ind)

print()
# Calcula a soma dos valore de VaR individuais
soma_var_relativo_individuais = sum(var_relativo_individuais)
print(f"Soma dos valores de VaR relativo individuais: {soma_var_relativo_individuais:.2f}")


print(f'\nEfeito de diversificação: {soma_var_relativo_individuais - var_relativo:.2f}')

matriz_unidade = np.ones((8, 8))
matriz_unidade

desvios_padrao_vetor = np.array(desvios_padrao).reshape(-1, 1)

pesos = pesos_otimos_exemplo

vol_port_corr_1 = np.sqrt((pesos * desvios_padrao_vetor).T.dot(matriz_unidade.dot(pesos * desvios_padrao_vetor)))
print(f'\nVolatilidade do portfólio com correlações perfeitas positivas: {vol_port_corr_1[0][0]*100:.4f}%')

desvio_padrao = vol_port_corr_1[0][0] # volatilidade do portfólio se correlações fossem iguais a +1 

var_relativo_nao_diversificado = var_delta_normal_iid(valor_inicial, nivel_confianca, desvio_padrao, media, periodos, relativo=1)
print(f'\nValor inicial: {valor_inicial}, VaR Rel. Não-diversificado - T = {periodos} e {nivel_confianca*100:.0f}% de conf.: {var_relativo_nao_diversificado:.2f}')


# In[ ]:





# In[ ]:





# In[25]:


df = sharpe_ratios.copy()

df = df.sort_values(by=["Setor", 252], ascending=[True, False])
top5_per_sector = df.groupby("Setor").head(3).reset_index(drop=False)

top5_per_sector.columns = ['Ticker', 'Perfomance', 'Setor', 'Nome']
top5_per_sector["Ticker"] = top5_per_sector["Ticker"].str.replace(".SA", "", regex=False)

# Export all data to a single worksheet
df.to_excel("setores_agrupados.xlsx", sheet_name="TodosSetores", index=False)


# In[ ]:





# In[ ]:




!jupyter nbconvert --to script "Untitled.ipynb"