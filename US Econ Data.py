import pandas as pd
import numpy as np
import yfinance as yf
from full_fred.fred import Fred
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import jinja2

print("Starting data collection process...")

fred = Fred('Data Collection/key.txt')
fred.set_api_key_file('Data Collection/key.txt')
print("FRED API connection established")

def daily(x):
    print(f"Downloading daily data for {x}...")
    dta = yf.download(tickers=x, period='max', interval='1d')
    dta[x] = dta['Close']
    dta = dta.drop(columns=['Open','High','Low','Close','Volume'])
    dta[x] = np.array(dta[x])
    return dta[x]

def data(x,y):
    print(f"Fetching {y} frequency data for {x}...")
    a = fred.get_series_df(x,frequency=y)
    a.value = a.value.replace('.',np.nan)
    a.value = a.value.ffill()
    a.index = a.date
    a = a.drop(columns=['date','realtime_start','realtime_end'])
    a.value = a.value.astype('float')
    return a

def fed_annual(x):
    print(f"Fetching annual data for {x}...")
    a = fred.get_series_df(x,frequency='a')
    a.value = a.value.replace('.',np.nan)
    a.value = a.value.ffill()
    a.index = pd.to_datetime(a.date)
    a = a.drop(columns=['date','realtime_start','realtime_end'])
    a.value = a.value.astype('float')
    return a

def growth(x):
    y = (x/x.shift(1))-1
    y = y.fillna(0)
    return y

def options_data(Ticker, nth_expiration_date):
    current_price = daily(Ticker).tail(1)
    ticker = yf.Ticker(Ticker)
    ticker_option_chain = ticker.option_chain(ticker.options[nth_expiration_date])
    return ticker, ticker_option_chain, current_price

def black_scholes(ticker, K, expiration_date, r, option_type='call'):
    S = daily(ticker).tail(1).values
    sigma = daily(ticker).std()
    current_date = datetime.now()
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    T = (expiration_date - current_date).days / 365.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Must be either 'call' or 'put'.")

    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else
             -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'option_price': option_price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def binomial_option_price(ticker, K, expiration_date, r, n, option_type='call'):
    S = daily(ticker).tail(1).values
    sigma = daily(ticker).std()
    current_date = datetime.now()
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    T = (expiration_date - current_date).days / 365.0

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    option_price_tree = np.zeros((n + 1, n + 1))
    
    for j in range(n + 1):
        if option_type == 'call':
            option_price_tree[n, j] = max(0, S * (u ** j) * (d ** (n - j)) - K)
        elif option_type == 'put':
            option_price_tree[n, j] = max(0, K - S * (u ** j) * (d ** (n - j)))

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if option_type == 'call':
                option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))
            elif option_type == 'put':
                option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))

    return option_price_tree[0, 0]

print("Processing FRED IDs...")
ids = ['gdp', 'm1v', 'm2v', 'GFDEBTN', 'RESPPANWW', 'QBPBSTAS',
       'TOTALSL', 'DRALACBS', 'PAYEMS', 'unrate', 'CIVPART',
       'CPIAUCSL','PCE','RSXFS','TOTALSA','JTSJOL','INDPRO','CSUSHPINSA'
       ,'IEABC','BOPGSTB','ATLSBUSRGEP','TTLCONS','QBPQYNTIY','H8B1058NCBCMG','TMBACBW027SBOG'
       ,'QBPBSTASSCUSTRSC','QBPQYNUMINST','WSHOMCB','DCPF3M','TOTBORR'
       ,'BAMLHYH0A0HYM2TRIV','BUSINV','TOTBUSSMSA','FRGSHPUSM649NCIS'
       ,'PETROLEUMD11','RHORUSQ156N','ACTLISCOUUS','HOSMEDUSM052N'
       ,'PATENTUSALLTOTAL','MNFCTRIRSA','MNFCTRIMSA','DTCDISA066MSFRBNY','RETAILIMSA'
       ,'RSCCAS','DFF','DCPF3M','DGS10','DGS1','DGS5','DGS30','DGS2','DGS1','DGS2'
       ,'DGS20','DGS3','DGS7','DSPIC96','DRCCLACBS','PSAVERT','DTB4WK','DTB3','PPIACO','GDPC1']

print(f"Processing {len(ids)} FRED IDs...")
df = pd.DataFrame()
for i in ids:
    print(f"Processing FRED ID: {i}")
    y = fred.get_a_series(i)
    x = y['seriess']
    a = pd.DataFrame(x).transpose()
    b = a[0]
    df[b.id] = b

df = df.transpose()
df = df.drop(columns=['seasonal_adjustment','popularity','units','realtime_start','realtime_end'])
df = df.sort_values(by='observation_start')
df = df.reset_index()
df = df.drop(columns='index')

ids = list(df.id)
names = list(df.title)

dta = pd.DataFrame()
for i in ids:
    y = fed_annual(i)
    dta[i] = y.value

labor_market = list(['PAYEMS','CIVPART','UNRATE','JTSJOL','ICSA','AWHAETP','CES0500000003','CE16OV','CLF16OV'])
rates = list(['DTB4WK','DTB3','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30','DFF','DCPF3M'])
production = list(['INDPRO','GDP','MNFCTRIMSA','FRGSHPUSM649NCIS','MNFCTRIRSA','PATENTUSALLTOTAL','PETROLEUMD11','DTCDISA066MSFRBNY'])
consumer_spending = list(['TOTALSA','RETAILIMSA','TOTBUSSMSA','BOPGSTB','RSCCAS','RSXFS','BUSINV','ATLSBUSRGEP','DSPIC96'])
consumer_debt = list(['TOTALSL','GFDEBTN','H8B1058NCBCMG','RESPPANWW','TMBACBW027SBOG','DRALACBS'])
housing = list(['RHORUSQ156N','CSUSHPINSA','TTLCONS','ACTLISCOUUS','EXHOSLUSM495S','HOSMEDUSM052N'])
prices = list(['CPIAUCSL','PCE','PPIACO'])
money = list(['M2','M2V'])
government_debt = list(['QBPBSTAS','QBPBSTASSCUSTRSC','WSHOMCB','GFDEBTN'])
trade = list(['FRGSHPUSM649NCIS','BOPGSTB','PETROLEUMD11','DTCDISA066MSFRBNY'])

print("Processing labor market data...")
lm = pd.DataFrame()
for i in labor_market:
    print(f"Processing labor market indicator: {i}")
    y = data(i,'m')
    lm[i] = y.value

lm['labor_demand'] = (lm['CLF16OV'] + lm.JTSJOL)/1000
lm['labor_supply'] = lm['CE16OV']/1000

labor_market = lm
labor_market.index = pd.to_datetime(labor_market.index)

labor_market['labor_difference'] = labor_market['labor_supply']-labor_market['labor_demand']
labor_market['payems_change'] = labor_market['PAYEMS']-labor_market['PAYEMS'].shift(1)
labor_market['awh_yoy'] = (labor_market['AWHAETP']/labor_market['AWHAETP'].shift(12)-1)*100
labor_market['ahe_yoy'] = (labor_market['CES0500000003']/labor_market['CES0500000003'].shift(12)-1)*100
labor_market.columns = ["All Employees, Total Nonfarm", "Labor Force Participation Rate", "Unemployment Rate", "Job Openings: Total Nonfarm","Initial Claims", "Average Weekly Hours", "Average Hourly Earnings","Civilial Labor Force Level","Employment Level","Labor Demand","Labor Supply","Labor Difference","Nonfarm Change","AWH YoY%","AHE YoY%"]

print("Processing rates data...")
r = pd.DataFrame()
for i in rates:
    print(f"Processing rate indicator: {i}")
    y = data(i,'d')
    r[i] = y.value
r.index = pd.to_datetime(r.index)

yields = r.drop(columns=['DFF','DCPF3M'])
yields.columns = ['1mo','3mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
yield_curve = yields.loc[r.index.max()]
yields.index = yields.index.strftime('%Y-%m-%d')

print("Processing sector data...")
sectors = ['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLC','XLU','^GSPC']
sectors_names = ['Cons. Discreat.','Cons. Staples','Energy','Financials',
                 'Health Care','Industrials','Materials','Real Estate','Tech',
                 'Comms','Utilitites','S&P 500']

print("Downloading sector data from Yahoo Finance...")
sectors_data = daily(sectors)
sectors_data.index = pd.to_datetime(sectors_data.index)

print("Processing corporate debt data...")
corp = list(['BAMLC1A0C13YEY','BAMLC2A0C35YEY','BAMLC3A0C57YEY','BAMLC4A0C710YEY','BAMLC7A0C1015YEY','BAMLC8A0C15PYEY','BAMLH0A0HYM2EY','BAMLC0A0CMEY'])
corp_debt = pd.DataFrame()
for i in corp:
    print(f"Processing corporate debt indicator: {i}")
    corp_debt[i] = data(i,'d')

corp_debt.index = pd.to_datetime(corp_debt.index)
corp_debt_y = corp_debt.groupby(pd.Grouper(freq='1YE')).mean()
corp_debt_std = corp_debt.rolling(21).std()

corp = corp_debt
corp.columns = ['1-3yrs','3-5yrs','5-7yrs','7-10yrs','10-15yrs','15+ yrs','High Yield Index','US Corp Index']
corp_curve = corp.drop(columns=['High Yield Index','US Corp Index'])
corp_curve = corp_curve.loc[corp_curve.index.max()]

print("Processing CPI indicators...")
indicators = list(['CPIUFDSL','CUSR0000SAF11','CUSR0000SEFV','CPIENGSL','CUSR0000SETB01','CUSR0000SEHE','CUSR0000SEHF','CUSR0000SEHF01','CUSR0000SACL1E','CUSR0000SAD','CUSR0000SAN','CUSR0000SASLE','CUSR0000SAH1','CUSR0000SAM2','CUSR0000SAS4','CPIAUCSL'])
cpi = pd.DataFrame()
for i in indicators:
    print(f"Processing CPI indicator: {i}")
    cpi[i] = data(i,'m')

cpi_yoy = (cpi / cpi.shift(12))-1

p = pd.DataFrame()
for i in prices:
    y = data(i,'m')
    p[i] = y.value

p['cpi_growth_yoy'] = p.CPIAUCSL / p.CPIAUCSL.shift(12) - 1
p['pce_growth_yoy'] = p.PCE / p.PCE.shift(12) - 1
p['ppi_growth_yoy'] = p.PPIACO / p.PPIACO.shift(12) - 1

consumer = cpi_yoy*100
consumer.columns = ["All items","Food","Food at home","Food away from home(1)","Energy","Gasoline","Fuel Oil","Energy services","Electricity","Commodities","Durable Goods","Non-Durable Goods","Service","Shelter","Medical","Transportation"]

prices = p*100
prices.index = pd.to_datetime(prices.index)
prices = prices.drop(columns=['CPIAUCSL','PCE','PPIACO'])
prices.columns = ['CPI YoY','PCE YoY','PPI YoY']

print("Processing S&P 500 data...")
sp = pd.DataFrame()
sp['close'] = daily('^GSPC')
print("Calculating S&P 500 technical indicators...")
sp['100ma'] = sp.close.rolling(100).mean()
sp['200ma'] = sp.close.rolling(200).mean()
sp['30mstd'] = sp.close.pct_change().rolling(30).std()*100
sp['60mstd'] = sp.close.pct_change().rolling(60).std()*100
sp['100mstd'] = sp.close.pct_change().rolling(100).std()*100
sp['200mstd'] = sp.close.pct_change().rolling(200).std()*100
sp['zscore 100ma'] = (sp.close - sp.close.mean())/sp.close.std()
sp['daily'] = sp.close.pct_change()*100
sp['mom'] = ((sp.close/sp.close.shift(30))-1)*100
sp['yoy'] = ((sp.close/sp.close.shift(365))-1)*100
sp['ytd'] = (sp.groupby(sp.index.year)['close'].transform(lambda x: x / x.iloc[0] - 1))*100
sp['vix'] = daily('^VIX')

print("Creating summary DataFrame...")
summary_rows = ['S&P 500','S&P 500 Daily Growth','30d Moving Sigma','VIX','1mo Bill','10yr Treasury']
summary = pd.DataFrame(index=summary_rows)

today_date = date.today()
today_date = today_date.strftime("%Y-%m-%d")

summary[today_date] = [
    sp.close.tail(1).values[0],
    sp.daily.tail(1).values[0],
    sp['30mstd'].tail(1).values[0],
    sp.vix.tail(1).values[0],
    yield_curve['1mo'],
    yield_curve['1yr']
]

summary['30 Mean'] = [sp.close.tail(30).mean(),
                   sp.daily.tail(20).mean(),
                   sp['30mstd'].tail(1).values[0],
                   sp.vix.tail(30).mean(),
                   yields['1mo'].tail(30).mean(),
                   yields['10yr'].tail(30).mean()
]

print("Generating plots...")
print("Creating S&P 500 plots...")
sp_plot = sp.tail(30)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

axes[0, 0].plot(sp_plot.index,sp_plot.close,'k-o')
axes[0, 0].plot(sp_plot.index,sp_plot['100ma'],'r')
axes[0, 0].plot(sp_plot.index,sp_plot['200ma'],'b')
axes[0, 0].set_title('Benchmark')
axes[0, 0].legend(['S&P 500','100ma','200ma'])

axes[0, 1].plot(sp_plot.index,sp_plot['30mstd'])
axes[0, 1].plot(sp_plot.index,sp_plot['100mstd'])
axes[0, 1].plot(sp_plot.index,sp_plot['200mstd'])
axes[0, 1].set_title('S&P 500 Moving Volatility')
axes[0, 1].legend(['30m','100m','200m'])

axes[0, 2].plot(sp_plot.index,sp_plot['daily'])
axes[0, 2].plot(sp_plot.index,sp_plot['mom'])
axes[0, 2].plot(sp_plot.index,sp_plot['yoy'])
axes[0, 2].plot(sp_plot.index,sp_plot['ytd'])
axes[0, 2].set_title('S&P 500 Performance')
axes[0, 2].legend(['Daily','MoM','YoY','YTD'])

axes[1, 0].plot(sp_plot.vix,'k-o')
axes[1, 0].set_title('S&P 500 VIX')

sns.set_style("whitegrid")
plt.tight_layout()

op_data = options_data("SPY",1)
options_dates = list(op_data[0].options)
options_dates = pd.to_datetime(options_dates)
for i, date in enumerate(options_dates):
    if date.month == 12:
        last_december_index = i

op_data = options_data("SPY",last_december_index)

option_chain = op_data[1]
axes[1, 1].plot(option_chain.calls.strike, option_chain.calls.volume,'go',label='Calls Volume')
axes[1, 1].plot(option_chain.puts.strike, option_chain.puts.volume,'mo',label='Puts Volume')
axes[1, 1].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
axes[1, 1].legend(loc='upper right')
axes[1, 1].set_xlabel('Strike Price')
axes[1, 1].set_ylabel('$')
axes[1, 1].set_title('Options Volume')
axes[1, 1].set_ylabel('Volume')

sns.set_style("whitegrid")
plt.tight_layout()

op_data = options_data("SPY",last_december_index)

option_chain = op_data[1]
axes[1, 2].plot(option_chain.calls.strike, option_chain.calls.lastPrice, 'ro', label='Calls Price')
axes[1, 2].plot(option_chain.puts.strike, option_chain.puts.lastPrice, 'bo', label='Puts Price')
axes[1, 2].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
axes[1, 2].legend(loc='upper right')
axes[1, 2].set_xlabel('Strike Price')
axes[1, 2].set_ylabel('$')

ax2 = axes[1, 2].twinx()
ax2.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility,'go',label='Calls IV')
ax2.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility,'mo',label='Puts IV')
ax2.legend(loc='lower right')
ax2.set_ylabel('Volume')

plt.tight_layout()
plt.savefig('Data Collection/spy_plots.png')
plt.show()

print("Creating sector correlation plot...")
plt.figure(figsize=(18,8))
plt.tight_layout()
plt.title('Equity Sectors Trailling Correlation')
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(12).corr()['^GSPC']*100,color='none', edgecolor='blue', linewidth=2)
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(6).corr()['^GSPC']*100,color='none', edgecolor='red', linewidth=2)
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(3).corr()['^GSPC']*100,color='none', edgecolor='green', linewidth=2)
plt.grid(True)
plt.legend(['12 Months','6 Months','3 Months'])
plt.savefig("Data Collection/sectors correlation.png")

print("Creating sector performance plot...")
plt.figure(figsize=(18,8))

plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-2])-1)*100 , color='none', edgecolor='blue', linewidth=2, label='1-day Growth')
plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-8])-1)*100 , color='none', edgecolor='red', linewidth=2, label='7-day Growth')
plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-31])-1)*100 , color='none', edgecolor='green', linewidth=2, label='30-day Growth')

plt.xlabel('Sectors')
plt.ylabel('Growth')
plt.title('Equity Sectors Perfomance')
plt.legend()

plt.tight_layout()
plt.savefig('Data Collection/equity sectors perfomance.png')

print("Creating sector options volatility plot...")
sns.set_style("whitegrid")
plt.tight_layout()

tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLC', 'XLU']
num_plots = len(tickers)
sectors_names = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Materials', 'Real Estate', 'Technology', 'Communication Services', 'Utilities']

num_rows = 3
num_cols = 4

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

axes = axes.flatten()

for ax, ticker, sector_name in zip(axes, tickers, sectors_names):
    option_chain = options_data(ticker, 1)[1]
    ax.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility, 'ro', label=f'{ticker} Calls')
    ax.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility, 'bo', label=f'{ticker} Puts')
    ax.axvline(x= options_data(ticker,1)[2].values  , color='k', linestyle='--', linewidth=1)
    ax.legend(loc='upper right')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(f'Options Volatility - {sector_name}')

plt.tight_layout()
plt.savefig('Data Collection/equity sectors options volatility.png')

print("Creating yield curve plots...")
yields_plot = yields.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

sns.set_style("whitegrid", {'axes.grid' : False})
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-6], marker='o', linestyle='-',color='black')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-1], marker='o', linestyle='-',color='blue')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-13], marker='o', linestyle='-',color='red')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-29], marker='o', linestyle='-',color='green')
axes[0].legend([yields.iloc[len(yields.index)-6].name,
                yields.iloc[len(yields.index)-1].name,
                yields.iloc[len(yields.index)-13].name,
                yields.iloc[len(yields.index)-29].name])
axes[0].set_title('Yield Curve')
axes[0].set_xlabel('Maturity')
axes[0].set_ylabel('Interest Rate')

sns.set_style("whitegrid")
axes[1].plot(yields_plot.index,yields_plot['1mo'],'-o')
axes[1].plot(yields_plot.index,yields_plot['1yr'],'-o')
axes[1].plot(yields_plot.index,yields_plot['10yr'],'-o')
axes[1].plot(yields_plot.index,yields_plot['30yr'],'-o')
axes[1].set_title('Yield Historical')
axes[1].legend(['1mo','1yr','10yr','30yr'])
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Interest Rate')

plt.tight_layout()
plt.savefig('Data Collection/treasury curve.png')

sns.set_style("whitegrid")
plt.tight_layout()
plt.figure(figsize=(10, 6))
sns.heatmap(yields_plot, annot=True, fmt=".2f", cmap="inferno")
plt.title('Yield Curve Trailling')
plt.ylabel('Date')
plt.savefig('Data Collection/yield curve trailling heatmap.png')

print("Creating CPI plots...")
consumer_plot = consumer.tail(12)
prices_plot = prices.tail(12)

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

sns.set_style("whitegrid")
sns.heatmap(consumer_plot, annot=True, fmt=".2f", cmap="inferno", ax=axes[0])
axes[0].set_title('CPI Categories')
axes[0].set_ylabel('Date')

sns.set_style("whitegrid")
axes[1].plot(prices_plot)
axes[1].set_title('Price Indexes YoY% Change')
axes[1].legend(['CPI','PCE','PPI'])

plt.tight_layout()
plt.savefig('Data Collection/cpi plots.png')
plt.show()

print("Creating labor market plots...")
labor_market_plot = labor_market.tail(24)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

axes[0, 0].plot(labor_market_plot.index, labor_market_plot['Nonfarm Change'], color='black', label='Payems Change')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Payems Change')
axes[0, 0].legend(['NonFarm Payroll'])

ax2 = axes[0, 0].twinx()
ax2.plot(labor_market_plot.index, labor_market_plot['Unemployment Rate'], color='green', label='Unemployment Rate')
ax2.set_ylabel('Unemployment Rate')
ax2.legend(['Unemployment Rate'])
axes[0, 0].set_title('Employment Indicators')

axes[0, 1].grid(True)
axes[0, 1].plot(labor_market_plot['Unemployment Rate'],'-ok')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(3).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(6).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(12).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].set_title('Unemployment Rate')
axes[0, 1].set_ylabel('%')
axes[0, 1].set_xlabel('Date')
axes[0, 1].legend(['Current Rate','3M MA','6M MA','12M MA'])

axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Demand'],'-o')
axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Supply'],'-o')
axes[1, 0].set_ylabel('Thousands of Persons')
ax2 = axes[1, 0].twinx()
ax2.plot(labor_market_plot.index, labor_market_plot['Labor Difference'],'k-o')
ax2.set_ylabel('Labor Difference')
axes[1, 0].set_title('Labor Forces')
axes[1, 0].legend(['Labor Demand','Labor Supply'])

axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Weekly Hours'],'b-o')
axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Hourly Earnings'],'g-o')
axes[1, 1].legend(['Average Weekly Hours','Average Hourly Earnings'])
ax2 = axes[1, 1].twinx()
ax2.plot(labor_market_plot.index, labor_market_plot['AWH YoY%'],'-o')
ax2.plot(labor_market_plot.index, labor_market_plot['AHE YoY%'],'-o')
axes[1, 1].legend(['Average Weekly Hours YoY%','Average Hourly Earnings YoY%'])
axes[1, 1].set_title('Weekly Earnings')

plt.tight_layout()
plt.savefig('Data Collection/labor indicators.png')
plt.show()

print("Generating HTML report...")
today = datetime.now().strftime("%Y-%m-%d @ %H:%M")
date_time = datetime.now().strftime("%Y-%m-%d")

context = {'date_time':today,
           'sp500_value':  "${:,.2f}".format(summary[date_time].iloc[0])
           ,'sp500_daily': "{:,.2f}%".format(summary[date_time].iloc[1])
           ,'sp500_30d_std': "{:,.2f}".format(summary[date_time].iloc[2])
           ,'vix': "{:,.2f}".format(summary[date_time].iloc[3])
           ,'tbill': "{:,.2f}".format(summary[date_time].iloc[4])
           ,'tbond': "{:,.2f}".format(summary[date_time].iloc[5])
           ,'yields_table' : yields_plot.tail(5).to_html()
           }

print("Loading template file...")
template_loader = jinja2.FileSystemLoader('Data Collection')
template_env = jinja2.Environment(loader=template_loader)

template = template_env.get_template("Econ Report Template.html")
print("Template loaded successfully")

output_text = template.render(context)

file_path = "Data Collection/Econ Report.html"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(output_text)
print(f"Report generated successfully at {file_path}")

print("Data collection and analysis complete!")