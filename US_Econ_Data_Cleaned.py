from io import BytesIO
import jinja2
import datetime
from datetime import datetime
from datetime import date
from datetime import datetime
#!/usr/bin/env python
# coding: utf-8
# 

# Libraries
# 

# In[1]:
# 

# 

# 

#import nasdaqdatalink
# 

# 

# FRED Key
# 

# In[2]:
# 

# 

# fred = Fred('key.txt')
fred = Fred('key.txt')
# fred.set_api_key_file('key.txt')
fred.set_api_key_file('key.txt')
# 

# 

# Functions
# 

# In[3]:
# 

# 

# def daily(x):
def daily(x):
# dta = yf.download(tickers=x,period='max',interval = '1d')
    dta = yf.download(tickers=x,period='max',interval = '1d')
# dta[x] = dta['Close']
    dta[x] = dta['Close']
# dta = dta.drop(columns=['Open','High','Low','Close','Adj Close','Volume'])
    dta = dta.drop(columns=['Open','High','Low','Close','Adj Close','Volume'])
# dta[x] = np.array(dta[x])
    dta[x] = np.array(dta[x])
# return dta[x]
    return dta[x]
# 

# def meta_data(x):
def meta_data(x):
# y = fred.get_a_series(x)
    y = fred.get_a_series(x)
# dta = pd.DataFrame(y['seriess']).transpose()
    dta = pd.DataFrame(y['seriess']).transpose()
# return dta
    return dta
# 

# def data(x,y):
def data(x,y):
# a = fred.get_series_df(x,frequency=y)
    a = fred.get_series_df(x,frequency=y)
# a.value = a.value.replace('.',np.nan)
    a.value = a.value.replace('.',np.nan)
# a.value = a.value.ffill()
    a.value = a.value.ffill()
# a.index = a.date
    a.index = a.date
# a = a.drop(columns=['date','realtime_start','realtime_end'])
    a = a.drop(columns=['date','realtime_start','realtime_end'])
# a.value = a.value.astype('float')
    a.value = a.value.astype('float')
# return a
    return a
# 

# def data2(x):
def data2(x):
# a = fred.get_series_df(x,frequency='d')
    a = fred.get_series_df(x,frequency='d')
# a.value = a.value.replace('.',np.nan)
    a.value = a.value.replace('.',np.nan)
# a.value = a.value.ffill()
    a.value = a.value.ffill()
# a.index = a.date
    a.index = a.date
# a = a.drop(columns=['date','realtime_start','realtime_end'])
    a = a.drop(columns=['date','realtime_start','realtime_end'])
# a.value = a.value.astype('float')
    a.value = a.value.astype('float')
# return a
    return a
# 

# def fed_annual(x):
def fed_annual(x):
# a = fred.get_series_df(x,frequency='a')
    a = fred.get_series_df(x,frequency='a')
# a.value = a.value.replace('.',np.nan)
    a.value = a.value.replace('.',np.nan)
# a.value = a.value.ffill()
    a.value = a.value.ffill()
# a.index = pd.to_datetime(a.date)
    a.index = pd.to_datetime(a.date)
# a = a.drop(columns=['date','realtime_start','realtime_end'])
    a = a.drop(columns=['date','realtime_start','realtime_end'])
# a.value = a.value.astype('float')
    a.value = a.value.astype('float')
# return a
    return a
# 

# def growth(x):
def growth(x):
# y = (x/x.shift(1))-1
    y = (x/x.shift(1))-1
# y = y.fillna(0)
    y = y.fillna(0)
# return y
    return y
# 

# def exp(x):
def exp(x):
# p = x
    p = x
# p = round(p,2)
    p = round(p,2)
# bins , counts = np.unique(p,return_counts=True)
    bins , counts = np.unique(p,return_counts=True)
# a = pd.DataFrame()
    a = pd.DataFrame()
# a['bins'] =bins
    a['bins'] =bins
# a['counts'] = counts/sum(counts)
    a['counts'] = counts/sum(counts)
# e = sum(a.bins * a.counts)
    e = sum(a.bins * a.counts)
# e = [e]
    e = [e]
# return a
    return a
# 

# def nasdaq(x):
def nasdaq(x):
# df = nasdaqdatalink.get(x)
    df = nasdaqdatalink.get(x)
# df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)
# return df
    return df
# 

# def options_data(Ticker, nth_expiration_date):
def options_data(Ticker, nth_expiration_date):
# """
    """
# Retrieves option chain data for a given ticker and expiration date.
    Retrieves option chain data for a given ticker and expiration date.
# 
    
# Parameters:
    Parameters:
# - Ticker (str): Ticker symbol of the stock.
    - Ticker (str): Ticker symbol of the stock.
# - nth_expiration_date (int): Index of the expiration date in the list of available expiration dates.
    - nth_expiration_date (int): Index of the expiration date in the list of available expiration dates.
# 
    
# Returns:
    Returns:
# - ticker: Ticker object.
    - ticker: Ticker object.
# - ticker_option_chain: Option chain data for the specified expiration date.
    - ticker_option_chain: Option chain data for the specified expiration date.
# - current_price: Current price of the underlying asset.
    - current_price: Current price of the underlying asset.
# """
    """
# current_price = daily(Ticker).tail(1)
    current_price = daily(Ticker).tail(1)
# ticker = yf.Ticker(Ticker)
    ticker = yf.Ticker(Ticker)
# ticker_option_chain = ticker.option_chain(ticker.options[nth_expiration_date])
    ticker_option_chain = ticker.option_chain(ticker.options[nth_expiration_date])
# return ticker, ticker_option_chain, current_price
    return ticker, ticker_option_chain, current_price
# 

# def plot_options(nth_expiration_date, strike_price):
def plot_options(nth_expiration_date, strike_price):
# """
    """
# Plots option data including strike prices, last prices, and implied volatilities for calls and puts.
    Plots option data including strike prices, last prices, and implied volatilities for calls and puts.
# 
    
# Parameters:
    Parameters:
# - nth_expiration_date (int): Index of the expiration date in the list of available expiration dates.
    - nth_expiration_date (int): Index of the expiration date in the list of available expiration dates.
# - strike_price (float): Strike price for the options.
    - strike_price (float): Strike price for the options.
# """
    """
# fig, ax1 = plt.subplots(figsize=(20, 10))
    fig, ax1 = plt.subplots(figsize=(20, 10))
# exp_date = options_data(nth_expiration_date)[1]
    exp_date = options_data(nth_expiration_date)[1]
# 

# plt.title(f"Options Last Price - Expiration {exp_date}")
    plt.title(f"Options Last Price - Expiration {exp_date}")
# ax1.plot(options_data(nth_expiration_date)[0].calls.strike, options_data(nth_expiration_date)[0].calls[p], 'ro')
    ax1.plot(options_data(nth_expiration_date)[0].calls.strike, options_data(nth_expiration_date)[0].calls[p], 'ro')
# ax1.axvline(x=options_data(nth_expiration_date)[2], color='k', linestyle='--', linewidth=1)
    ax1.axvline(x=options_data(nth_expiration_date)[2], color='k', linestyle='--', linewidth=1)
# ax1.legend(['Calls', 'Current Price'], loc='upper left')
    ax1.legend(['Calls', 'Current Price'], loc='upper left')
# 
    
# ax2 = ax1.twinx()
    ax2 = ax1.twinx()
# ax2.plot(options_data(nth_expiration_date)[0].puts.strike, options_data(nth_expiration_date)[0].puts[p], 'o')
    ax2.plot(options_data(nth_expiration_date)[0].puts.strike, options_data(nth_expiration_date)[0].puts[p], 'o')
# ax2.legend(['Puts'], loc='upper right')
    ax2.legend(['Puts'], loc='upper right')
# 
    
# 
    
# def black_scholes(ticker, K, expiration_date, r, option_type='call'):
def black_scholes(ticker, K, expiration_date, r, option_type='call'):
# """
    """
# Calculate the price and Greeks of a European option using the Black-Scholes model.
    Calculate the price and Greeks of a European option using the Black-Scholes model.
# 

# Parameters:
    Parameters:
# ticker (str): Ticker symbol of the underlying stock
        ticker (str): Ticker symbol of the underlying stock
# K (float): Strike price
        K (float): Strike price
# expiration_date (str): Expiration date of the option in the format 'YYYY-MM-DD'
        expiration_date (str): Expiration date of the option in the format 'YYYY-MM-DD'
# r (float): Risk-free interest rate (annualized, as a decimal)
        r (float): Risk-free interest rate (annualized, as a decimal)
# option_type (str): Type of option, either 'call' or 'put' (default is 'call')
        option_type (str): Type of option, either 'call' or 'put' (default is 'call')
# 

# Returns:
    Returns:
# dict: A dictionary containing the theoretical price and Greeks of the European option
        dict: A dictionary containing the theoretical price and Greeks of the European option
# """
    """
    # Get latest stock price
# S = daily(ticker).tail(1).values
    S = daily(ticker).tail(1).values
# 

    # Calculate volatility (sigma)
# sigma = daily(ticker).std()
    sigma = daily(ticker).std()
# 

    # Calculate time until expiration
# current_date = datetime.now()
    current_date = datetime.now()
# expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
# T = (expiration_date - current_date).days / 365.0  # Time to expiration in years
    T = (expiration_date - current_date).days / 365.0  # Time to expiration in years
# 

# d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# d2 = d1 - sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
# 

    # Calculate option price
# if option_type == 'call':
    if option_type == 'call':
# option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
# elif option_type == 'put':
    elif option_type == 'put':
# option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# else:
    else:
# raise ValueError("Invalid option_type. Must be either 'call' or 'put'.")
        raise ValueError("Invalid option_type. Must be either 'call' or 'put'.")
# 

    # Calculate Greeks
# delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
# gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
# theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
# - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else
             - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else
# -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
# + r * K * np.exp(-r * T) * norm.cdf(-d2))
             + r * K * np.exp(-r * T) * norm.cdf(-d2))
# vega = S * np.sqrt(T) * norm.pdf(d1)
    vega = S * np.sqrt(T) * norm.pdf(d1)
# rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
# 

# return {
    return {
# 'option_price': option_price,
        'option_price': option_price,
# 'delta': delta,
        'delta': delta,
# 'gamma': gamma,
        'gamma': gamma,
# 'theta': theta,
        'theta': theta,
# 'vega': vega,
        'vega': vega,
# 'rho': rho
        'rho': rho
# }
    }
# 

# def binomial_option_price(ticker, K, expiration_date, r, n, option_type='call'):
def binomial_option_price(ticker, K, expiration_date, r, n, option_type='call'):
# """
    """
# Calculate the price of an American option using the Binomial Options Pricing Model.
    Calculate the price of an American option using the Binomial Options Pricing Model.
# 

# Parameters:
    Parameters:
# ticker (str): Ticker symbol of the underlying stock
        ticker (str): Ticker symbol of the underlying stock
# K (float): Strike price
        K (float): Strike price
# expiration_date (str): Expiration date of the option in the format 'YYYY-MM-DD'
        expiration_date (str): Expiration date of the option in the format 'YYYY-MM-DD'
# r (float): Risk-free interest rate (annualized, as a decimal)
        r (float): Risk-free interest rate (annualized, as a decimal)
# n (int): Number of time steps in the binomial tree
        n (int): Number of time steps in the binomial tree
# option_type (str): Type of option, either 'call' or 'put' (default is 'call')
        option_type (str): Type of option, either 'call' or 'put' (default is 'call')
# 

# Returns:
    Returns:
# float: Theoretical price of the American option
        float: Theoretical price of the American option
# """
    """
    # Get latest stock price
# S = daily(ticker).tail(1).values
    S = daily(ticker).tail(1).values
# 

    # Calculate volatility (sigma)
# sigma = daily(ticker).std()
    sigma = daily(ticker).std()
# 
    
    # Calculate time until expiration
# current_date = datetime.now()
    current_date = datetime.now()
# expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
# T = (expiration_date - current_date).days / 365.0  # Time to expiration in years
    T = (expiration_date - current_date).days / 365.0  # Time to expiration in years
# 

# dt = T / n  # Time step
    dt = T / n  # Time step
# u = np.exp(sigma * np.sqrt(dt))  # Up factor
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
# d = 1 / u  # Down factor
    d = 1 / u  # Down factor
# p = (np.exp(r * dt) - d) / (u - d)  # Probability of up move
    p = (np.exp(r * dt) - d) / (u - d)  # Probability of up move
# 

    # Initialize the option price tree
# option_price_tree = np.zeros((n + 1, n + 1))
    option_price_tree = np.zeros((n + 1, n + 1))
# 
    
    # Calculate option prices at expiration
# for j in range(n + 1):
    for j in range(n + 1):
# if option_type == 'call':
        if option_type == 'call':
# option_price_tree[n, j] = max(0, S * (u ** j) * (d ** (n - j)) - K)
            option_price_tree[n, j] = max(0, S * (u ** j) * (d ** (n - j)) - K)
# elif option_type == 'put':
        elif option_type == 'put':
# option_price_tree[n, j] = max(0, K - S * (u ** j) * (d ** (n - j)))
            option_price_tree[n, j] = max(0, K - S * (u ** j) * (d ** (n - j)))
# 

    # Calculate option prices at earlier nodes
# for i in range(n - 1, -1, -1):
    for i in range(n - 1, -1, -1):
# for j in range(i + 1):
        for j in range(i + 1):
# if option_type == 'call':
            if option_type == 'call':
# option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))
                option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))
# elif option_type == 'put':
            elif option_type == 'put':
# option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))
                option_price_tree[i, j] = max(0, np.exp(-r * dt) * (p * option_price_tree[i + 1, j] + (1 - p) * option_price_tree[i + 1, j + 1]))
# 

# return option_price_tree[0, 0]
    return option_price_tree[0, 0]
# 

# 

# In[4]:
# 

# 

#Fed Data id codes
# ids = ['gdp', 'm1v', 'm2v', 'GFDEBTN', 'RESPPANWW', 'QBPBSTAS',
ids = ['gdp', 'm1v', 'm2v', 'GFDEBTN', 'RESPPANWW', 'QBPBSTAS',
# 'TOTALSL', 'DRALACBS', 'PAYEMS', 'unrate', 'CIVPART',
       'TOTALSL', 'DRALACBS', 'PAYEMS', 'unrate', 'CIVPART',
# 'CPIAUCSL','PCE','RSXFS','TOTALSA','JTSJOL','INDPRO','CSUSHPINSA'
       'CPIAUCSL','PCE','RSXFS','TOTALSA','JTSJOL','INDPRO','CSUSHPINSA'
# ,'IEABC','BOPGSTB','ATLSBUSRGEP','TTLCONS','QBPQYNTIY','H8B1058NCBCMG','TMBACBW027SBOG'
       ,'IEABC','BOPGSTB','ATLSBUSRGEP','TTLCONS','QBPQYNTIY','H8B1058NCBCMG','TMBACBW027SBOG'
# ,'QBPBSTASSCUSTRSC','QBPQYNUMINST','WSHOMCB','DCPF3M','TOTBORR'
       ,'QBPBSTASSCUSTRSC','QBPQYNUMINST','WSHOMCB','DCPF3M','TOTBORR'
# ,'BAMLHYH0A0HYM2TRIV','BUSINV','TOTBUSSMSA','FRGSHPUSM649NCIS'
       ,'BAMLHYH0A0HYM2TRIV','BUSINV','TOTBUSSMSA','FRGSHPUSM649NCIS'
# ,'PETROLEUMD11','RHORUSQ156N','ACTLISCOUUS','HOSMEDUSM052N'
       ,'PETROLEUMD11','RHORUSQ156N','ACTLISCOUUS','HOSMEDUSM052N'
# ,'PATENTUSALLTOTAL','MNFCTRIRSA','MNFCTRIMSA','DTCDISA066MSFRBNY','RETAILIMSA'
       ,'PATENTUSALLTOTAL','MNFCTRIRSA','MNFCTRIMSA','DTCDISA066MSFRBNY','RETAILIMSA'
# ,'RSCCAS','DFF','DCPF3M','DGS10','DGS1','DGS5','DGS30','DGS2','DGS1','DGS2'
       ,'RSCCAS','DFF','DCPF3M','DGS10','DGS1','DGS5','DGS30','DGS2','DGS1','DGS2'
# ,'DGS20','DGS3','DGS7','DSPIC96','DRCCLACBS','PSAVERT','DTB4WK','DTB3','PPIACO','GDPC1']
       ,'DGS20','DGS3','DGS7','DSPIC96','DRCCLACBS','PSAVERT','DTB4WK','DTB3','PPIACO','GDPC1']
# 

# 

# In[5]:
# 

# 

# df = pd.DataFrame()
df = pd.DataFrame()
# 

# for i in ids:
for i in ids:
# y = fred.get_a_series(i)
    y = fred.get_a_series(i)
# x = y['seriess']
    x = y['seriess']
# a = pd.DataFrame(x).transpose()
    a = pd.DataFrame(x).transpose()
# b = a[0]
    b = a[0]
# df[b.id] = b
    df[b.id] = b
# 

# df = df.transpose()
df = df.transpose()
# df = df.drop(columns=['seasonal_adjustment','popularity','units','realtime_start','realtime_end'])
df = df.drop(columns=['seasonal_adjustment','popularity','units','realtime_start','realtime_end'])
# df = df.sort_values(by='observation_start')
df = df.sort_values(by='observation_start')
# df = df.reset_index()
df = df.reset_index()
# 

# df = df.drop(columns='index')
df = df.drop(columns='index')
#df.to_excel('meta_data.xlsx')
# 

# 

# In[6]:
# 

# 

# ids = list(df.id)
ids = list(df.id)
# names = list(df.title)
names = list(df.title)
# 

# 

# In[7]:
# 

# 

# dta = pd.DataFrame()
dta = pd.DataFrame()
# 

# for i in ids:
for i in ids:
# y = fed_annual(i)
    y = fed_annual(i)
# dta[i] = y.value
    dta[i] = y.value
# 

# 

# FRED API Data Calling
# 

# In[8]:
# 

# 

#Labor Market
# labor_market = list(['PAYEMS','CIVPART','UNRATE','JTSJOL','ICSA','AWHAETP','CES0500000003','CE16OV','CLF16OV'])
labor_market = list(['PAYEMS','CIVPART','UNRATE','JTSJOL','ICSA','AWHAETP','CES0500000003','CE16OV','CLF16OV'])
# 

#Rates
# rates = list(['DTB4WK','DTB3','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30','DFF','DCPF3M'])
rates = list(['DTB4WK','DTB3','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30','DFF','DCPF3M'])
# 

#Production
# production = list(['GDP','GDPC1','GDI','A261RX1Q020SBEA'])
production = list(['GDP','GDPC1','GDI','A261RX1Q020SBEA'])
# 

#Consumer Spending
# consumer_spending = list(['TOTALSA','RETAILIMSA','TOTBUSSMSA','BOPGSTB','RSCCAS','RSXFS','BUSINV','ATLSBUSRGEP','DSPIC96'])
consumer_spending = list(['TOTALSA','RETAILIMSA','TOTBUSSMSA','BOPGSTB','RSCCAS','RSXFS','BUSINV','ATLSBUSRGEP','DSPIC96'])
# 

#Consumer Debt
# consumer_debt = list(['TOTALSL','GFDEBTN','H8B1058NCBCMG','RESPPANWW','TMBACBW027SBOG','DRALACBS'])
consumer_debt = list(['TOTALSL','GFDEBTN','H8B1058NCBCMG','RESPPANWW','TMBACBW027SBOG','DRALACBS'])
# 

#Housing
# housing = list(['RHORUSQ156N','CSUSHPINSA','TTLCONS','ACTLISCOUUS','EXHOSLUSM495S','HOSMEDUSM052N'])
housing = list(['RHORUSQ156N','CSUSHPINSA','TTLCONS','ACTLISCOUUS','EXHOSLUSM495S','HOSMEDUSM052N'])
# 

#Monetary & Prices
# prices = list(['CPIAUCSL','PCE','PPIACO'])
prices = list(['CPIAUCSL','PCE','PPIACO'])
# money = list(['M2','M2V'])
money = list(['M2','M2V'])
# 

#Government Debt
# government_debt = list(['QBPBSTAS','QBPBSTASSCUSTRSC','WSHOMCB','GFDEBTN'])
government_debt = list(['QBPBSTAS','QBPBSTASSCUSTRSC','WSHOMCB','GFDEBTN'])
# 

#Trade
# trade = list(['FRGSHPUSM649NCIS','BOPGSTB','PETROLEUMD11','DTCDISA066MSFRBNY'])
trade = list(['FRGSHPUSM649NCIS','BOPGSTB','PETROLEUMD11','DTCDISA066MSFRBNY'])
# 

# 

# In[9]:
# 

# 

# prod = pd.DataFrame()
prod = pd.DataFrame()
# 

# for i in production:
for i in production:
# y = data(i,'q')
    y = data(i,'q')
# prod[i] = y.value
    prod[i] = y.value
# 

# 

# In[10]:
# 

# 

# lm = pd.DataFrame()
lm = pd.DataFrame()
# 

# for i in labor_market:
for i in labor_market:
# y = data(i,'m')
    y = data(i,'m')
# lm[i] = y.value
    lm[i] = y.value
# 

# lm['labor_demand'] = (lm['CLF16OV'] + lm.JTSJOL)/1000
lm['labor_demand'] = (lm['CLF16OV'] + lm.JTSJOL)/1000
# lm['labor_supply'] = lm['CE16OV']/1000
lm['labor_supply'] = lm['CE16OV']/1000
# 

#Labor Market
# labor_market = lm
labor_market = lm
# labor_market.index = pd.to_datetime(labor_market.index)
labor_market.index = pd.to_datetime(labor_market.index)
# 

# labor_market['labor_difference'] = labor_market['labor_supply']-labor_market['labor_demand']
labor_market['labor_difference'] = labor_market['labor_supply']-labor_market['labor_demand']
# labor_market['payems_change'] = labor_market['PAYEMS']-labor_market['PAYEMS'].shift(1)
labor_market['payems_change'] = labor_market['PAYEMS']-labor_market['PAYEMS'].shift(1)
# labor_market['awh_yoy'] = (labor_market['AWHAETP']/labor_market['AWHAETP'].shift(12)-1)*100
labor_market['awh_yoy'] = (labor_market['AWHAETP']/labor_market['AWHAETP'].shift(12)-1)*100
# labor_market['ahe_yoy'] = (labor_market['CES0500000003']/labor_market['CES0500000003'].shift(12)-1)*100
labor_market['ahe_yoy'] = (labor_market['CES0500000003']/labor_market['CES0500000003'].shift(12)-1)*100
# labor_market.columns = ["All Employees, Total Nonfarm", "Labor Force Participation Rate", "Unemployment Rate", "Job Openings: Total Nonfarm","Initial Claims", "Average Weekly Hours", "Average Hourly Earnings","Civilial Labor Force Level","Employment Level","Labor Demand","Labor Supply","Labor Difference","Nonfarm Change","AWH YoY%","AHE YoY%"]
labor_market.columns = ["All Employees, Total Nonfarm", "Labor Force Participation Rate", "Unemployment Rate", "Job Openings: Total Nonfarm","Initial Claims", "Average Weekly Hours", "Average Hourly Earnings","Civilial Labor Force Level","Employment Level","Labor Demand","Labor Supply","Labor Difference","Nonfarm Change","AWH YoY%","AHE YoY%"]
# 

# 

# 

# In[11]:
# 

# 

# r = pd.DataFrame()
r = pd.DataFrame()
# 

# for i in rates:
for i in rates:
# y = data(i,'d')
    y = data(i,'d')
# r[i] = y.value
    r[i] = y.value
# r.index = pd.to_datetime(r.index)
r.index = pd.to_datetime(r.index)
# 

# 

#Treasury Yields table
# yields = r.drop(columns=['DFF','DCPF3M'])
yields = r.drop(columns=['DFF','DCPF3M'])
# yields.columns = ['1mo','3mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
yields.columns = ['1mo','3mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
# yield_curve = yields.loc[r.index.max()]
yield_curve = yields.loc[r.index.max()]
# yields.index = yields.index.strftime('%Y-%m-%d')
yields.index = yields.index.strftime('%Y-%m-%d')
# 

# 

# In[12]:
# 

# 

#US Sectors
#Tickers
# sectors = ['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLC','XLU','^GSPC']
sectors = ['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLC','XLU','^GSPC']
# sectors_names = ['Cons. Discreat.','Cons. Staples','Energy','Financials',
sectors_names = ['Cons. Discreat.','Cons. Staples','Energy','Financials',
# 'Health Care','Industrials','Materials','Real Estate','Tech',
                 'Health Care','Industrials','Materials','Real Estate','Tech',
# 'Comms','Utilitites','S&P 500']
                 'Comms','Utilitites','S&P 500']
# 

# 

# In[13]:
# 

# 

# sectors_data = daily(sectors)
sectors_data = daily(sectors)
# sectors_data.index = pd.to_datetime(sectors_data.index)
sectors_data.index = pd.to_datetime(sectors_data.index)
#sectors_data = sectors_data.groupby(pd.Grouper(freq='1ME')).last()
#sectors_data_growth = sectors_data.pct_change().fillna(0)
# 

# 

# In[14]:
# 

# 

#Corporate debt 
# 

# corp = list(['BAMLC1A0C13YEY','BAMLC2A0C35YEY','BAMLC3A0C57YEY','BAMLC4A0C710YEY','BAMLC7A0C1015YEY','BAMLC8A0C15PYEY','BAMLH0A0HYM2EY','BAMLC0A0CMEY'])
corp = list(['BAMLC1A0C13YEY','BAMLC2A0C35YEY','BAMLC3A0C57YEY','BAMLC4A0C710YEY','BAMLC7A0C1015YEY','BAMLC8A0C15PYEY','BAMLH0A0HYM2EY','BAMLC0A0CMEY'])
# 

# corp_debt = pd.DataFrame()
corp_debt = pd.DataFrame()
# 

# for i in corp:
for i in corp:
# corp_debt[i] = data(i,'d')
    corp_debt[i] = data(i,'d')
# 

# corp_debt.index = pd.to_datetime(corp_debt.index)
corp_debt.index = pd.to_datetime(corp_debt.index)
# 

# corp_debt_y = corp_debt.groupby(pd.Grouper(freq='1YE')).mean()
corp_debt_y = corp_debt.groupby(pd.Grouper(freq='1YE')).mean()
# corp_debt_std = corp_debt.rolling(21).std()
corp_debt_std = corp_debt.rolling(21).std()
# 

#Corporate Debt Yields
# corp = corp_debt
corp = corp_debt
# corp.columns = ['1-3yrs','3-5yrs','5-7yrs','7-10yrs','10-15yrs','15+ yrs','High Yield Index','US Corp Index']
corp.columns = ['1-3yrs','3-5yrs','5-7yrs','7-10yrs','10-15yrs','15+ yrs','High Yield Index','US Corp Index']
# corp_curve = corp.drop(columns=['High Yield Index','US Corp Index'])
corp_curve = corp.drop(columns=['High Yield Index','US Corp Index'])
# corp_curve = corp_curve.loc[corp_curve.index.max()]
corp_curve = corp_curve.loc[corp_curve.index.max()]
# 

# 

# In[15]:
# 

# 

#CPI indicators
# indicators = list(['CPIUFDSL','CUSR0000SAF11','CUSR0000SEFV','CPIENGSL','CUSR0000SETB01','CUSR0000SEHE','CUSR0000SEHF','CUSR0000SEHF01','CUSR0000SACL1E','CUSR0000SAD','CUSR0000SAN','CUSR0000SASLE','CUSR0000SAH1','CUSR0000SAM2','CUSR0000SAS4','CPIAUCSL'])
indicators = list(['CPIUFDSL','CUSR0000SAF11','CUSR0000SEFV','CPIENGSL','CUSR0000SETB01','CUSR0000SEHE','CUSR0000SEHF','CUSR0000SEHF01','CUSR0000SACL1E','CUSR0000SAD','CUSR0000SAN','CUSR0000SASLE','CUSR0000SAH1','CUSR0000SAM2','CUSR0000SAS4','CPIAUCSL'])
# 

# cpi = pd.DataFrame()
cpi = pd.DataFrame()
# for i in indicators:
for i in indicators:
# cpi[i] = data(i,'m')
    cpi[i] = data(i,'m')
# 

# cpi_yoy = (cpi / cpi.shift(12))-1
cpi_yoy = (cpi / cpi.shift(12))-1
# 

# p = pd.DataFrame()
p = pd.DataFrame()
# 

# for i in prices:
for i in prices:
# y = data(i,'m')
    y = data(i,'m')
# p[i] = y.value
    p[i] = y.value
# 

# p['cpi_growth_yoy'] = p.CPIAUCSL / p.CPIAUCSL.shift(12) - 1
p['cpi_growth_yoy'] = p.CPIAUCSL / p.CPIAUCSL.shift(12) - 1
# p['pce_growth_yoy'] = p.PCE / p.PCE.shift(12) - 1
p['pce_growth_yoy'] = p.PCE / p.PCE.shift(12) - 1
# p['ppi_growth_yoy'] = p.PPIACO / p.PPIACO.shift(12) - 1
p['ppi_growth_yoy'] = p.PPIACO / p.PPIACO.shift(12) - 1
# 

#CPI
# consumer = cpi_yoy*100
consumer = cpi_yoy*100
# consumer.columns = ["All items","Food","Food at home","Food away from home(1)","Energy","Gasoline","Fuel Oil","Energy services","Electricity","Commodities","Durable Goods","Non-Durable Goods","Service","Shelter","Medical","Transportation"]
consumer.columns = ["All items","Food","Food at home","Food away from home(1)","Energy","Gasoline","Fuel Oil","Energy services","Electricity","Commodities","Durable Goods","Non-Durable Goods","Service","Shelter","Medical","Transportation"]
# 

#Prices Index
# prices = p*100
prices = p*100
# prices.index = pd.to_datetime(prices.index)
prices.index = pd.to_datetime(prices.index)
# prices = prices.drop(columns=['CPIAUCSL','PCE','PPIACO'])
prices = prices.drop(columns=['CPIAUCSL','PCE','PPIACO'])
# prices.columns = ['CPI YoY','PCE YoY','PPI YoY']
prices.columns = ['CPI YoY','PCE YoY','PPI YoY']
# 

# 

# Data Tables Formated
# 

# In[16]:
# 

# 

#Summary
#S&P 500
# sp = pd.DataFrame()
sp = pd.DataFrame()
# sp['close'] = daily('^GSPC')
sp['close'] = daily('^GSPC')
# sp['100ma'] = sp.close.rolling(100).mean()
sp['100ma'] = sp.close.rolling(100).mean()
# sp['200ma'] = sp.close.rolling(200).mean()
sp['200ma'] = sp.close.rolling(200).mean()
# sp['30mstd'] = sp.close.pct_change().rolling(30).std()*100
sp['30mstd'] = sp.close.pct_change().rolling(30).std()*100
# sp['60mstd'] = sp.close.pct_change().rolling(60).std()*100
sp['60mstd'] = sp.close.pct_change().rolling(60).std()*100
# sp['100mstd'] = sp.close.pct_change().rolling(100).std()*100
sp['100mstd'] = sp.close.pct_change().rolling(100).std()*100
# sp['200mstd'] = sp.close.pct_change().rolling(200).std()*100
sp['200mstd'] = sp.close.pct_change().rolling(200).std()*100
# sp['zscore 100ma'] = (sp.close - sp.close.mean())/sp.close.std()
sp['zscore 100ma'] = (sp.close - sp.close.mean())/sp.close.std()
# sp['daily'] = sp.close.pct_change()*100
sp['daily'] = sp.close.pct_change()*100
# sp['mom'] = ((sp.close/sp.close.shift(30))-1)*100
sp['mom'] = ((sp.close/sp.close.shift(30))-1)*100
# sp['yoy'] = ((sp.close/sp.close.shift(365))-1)*100
sp['yoy'] = ((sp.close/sp.close.shift(365))-1)*100
# sp['ytd'] = (sp.groupby(sp.index.year)['close'].transform(lambda x: x / x.iloc[0] - 1))*100
sp['ytd'] = (sp.groupby(sp.index.year)['close'].transform(lambda x: x / x.iloc[0] - 1))*100
# sp['vix'] = daily('^VIX')
sp['vix'] = daily('^VIX')
# 

# 

# In[17]:
# 

# 

#Summary
# summary_rows = ['S&P 500','S&P 500 Daily Growth','30d Moving Sigma','VIX','1mo Bill','10yr Treasury']
summary_rows = ['S&P 500','S&P 500 Daily Growth','30d Moving Sigma','VIX','1mo Bill','10yr Treasury']
# summary = pd.DataFrame(index=summary_rows)
summary = pd.DataFrame(index=summary_rows)
# 

# today_date = date.today()
today_date = date.today()
# today_date = today_date.strftime("%Y-%m-%d")
today_date = today_date.strftime("%Y-%m-%d")
# today_date
today_date
# 

# 

# summary[today_date] = [
summary[today_date] = [
# sp.close.tail(1).values[0],
    sp.close.tail(1).values[0],
# sp.daily.tail(1).values[0],
    sp.daily.tail(1).values[0],
# sp['30mstd'].tail(1).values[0],
    sp['30mstd'].tail(1).values[0],
# sp.vix.tail(1).values[0],
    sp.vix.tail(1).values[0],
# yield_curve['1mo'],
    yield_curve['1mo'],
# yield_curve['1yr']
    yield_curve['1yr']
# ]
]
# 

# summary['30 Mean'] = [sp.close.tail(30).mean(),
summary['30 Mean'] = [sp.close.tail(30).mean(),
# sp.daily.tail(20).mean(),
                   sp.daily.tail(20).mean(),
# sp['30mstd'].tail(1).values[0],
                   sp['30mstd'].tail(1).values[0],
# sp.vix.tail(30).mean(),
                   sp.vix.tail(30).mean(),
# yields['1mo'].tail(30).mean(),
                   yields['1mo'].tail(30).mean(),
# yields['10yr'].tail(30).mean()
                   yields['10yr'].tail(30).mean()
# 

# ]
]
# 

# 

# In[18]:
# 

# 

#S&P Plots
# sp_plot = sp.tail(30)
sp_plot = sp.tail(30)
# 

# 

# 

# 

# Create a figure and axes for the grid
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# 

# Plot 1: S&P Plots
# axes[0, 0].plot(sp_plot.index,sp_plot.close,'k-o')
axes[0, 0].plot(sp_plot.index,sp_plot.close,'k-o')
# axes[0, 0].plot(sp_plot.index,sp_plot['100ma'],'r')
axes[0, 0].plot(sp_plot.index,sp_plot['100ma'],'r')
# axes[0, 0].plot(sp_plot.index,sp_plot['200ma'],'b')
axes[0, 0].plot(sp_plot.index,sp_plot['200ma'],'b')
# axes[0, 0].set_title('Benchmark')
axes[0, 0].set_title('Benchmark')
# axes[0, 0].legend(['S&P 500','100ma','200ma'])
axes[0, 0].legend(['S&P 500','100ma','200ma'])
# 

# Plot 2: S&P 500 Moving Volatility
# axes[0, 1].plot(sp_plot.index,sp_plot['30mstd'])
axes[0, 1].plot(sp_plot.index,sp_plot['30mstd'])
# axes[0, 1].plot(sp_plot.index,sp_plot['100mstd'])
axes[0, 1].plot(sp_plot.index,sp_plot['100mstd'])
# axes[0, 1].plot(sp_plot.index,sp_plot['200mstd'])
axes[0, 1].plot(sp_plot.index,sp_plot['200mstd'])
# axes[0, 1].set_title('S&P 500 Moving Volatility')
axes[0, 1].set_title('S&P 500 Moving Volatility')
# axes[0, 1].legend(['30m','100m','200m'])
axes[0, 1].legend(['30m','100m','200m'])
# 

# Plot 3: S&P 500 Performance
# axes[0, 2].plot(sp_plot.index,sp_plot['daily'])
axes[0, 2].plot(sp_plot.index,sp_plot['daily'])
# axes[0, 2].plot(sp_plot.index,sp_plot['mom'])
axes[0, 2].plot(sp_plot.index,sp_plot['mom'])
# axes[0, 2].plot(sp_plot.index,sp_plot['yoy'])
axes[0, 2].plot(sp_plot.index,sp_plot['yoy'])
# axes[0, 2].plot(sp_plot.index,sp_plot['ytd'])
axes[0, 2].plot(sp_plot.index,sp_plot['ytd'])
# axes[0, 2].set_title('S&P 500 Performance')
axes[0, 2].set_title('S&P 500 Performance')
# axes[0, 2].legend(['Daily','MoM','YoY','YTD'])
axes[0, 2].legend(['Daily','MoM','YoY','YTD'])
# 

# Plot 4: S&P 500 VIX
# axes[1, 0].plot(sp_plot.vix,'k-o')
axes[1, 0].plot(sp_plot.vix,'k-o')
# axes[1, 0].set_title('S&P 500 VIX')
axes[1, 0].set_title('S&P 500 VIX')
# 

# Plot 5: Options Pricing and Volume
# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# plt.tight_layout()
plt.tight_layout()
# 

# op_data = options_data("SPY",1)
op_data = options_data("SPY",1)
# options_dates = list(op_data[0].options)
options_dates = list(op_data[0].options)
# options_dates = pd.to_datetime(options_dates)
options_dates = pd.to_datetime(options_dates)
# for i, date in enumerate(options_dates):
for i, date in enumerate(options_dates):
# if date.month == 12:
    if date.month == 12:
# last_december_index = i
        last_december_index = i
# 

# op_data = options_data("SPY",last_december_index)
op_data = options_data("SPY",last_december_index)
# 

# option_chain = op_data[1]
option_chain = op_data[1]
# axes[1, 1].plot(option_chain.calls.strike, option_chain.calls.volume,'go',label='Calls Volume')
axes[1, 1].plot(option_chain.calls.strike, option_chain.calls.volume,'go',label='Calls Volume')
# axes[1, 1].plot(option_chain.puts.strike, option_chain.puts.volume,'mo',label='Puts Volume')
axes[1, 1].plot(option_chain.puts.strike, option_chain.puts.volume,'mo',label='Puts Volume')
# axes[1, 1].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
axes[1, 1].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
# axes[1, 1].legend(loc='upper right')
axes[1, 1].legend(loc='upper right')
# axes[1, 1].set_xlabel('Strike Price')
axes[1, 1].set_xlabel('Strike Price')
# axes[1, 1].set_ylabel('$')
axes[1, 1].set_ylabel('$')
# axes[1, 1].set_title('Options Volume')
axes[1, 1].set_title('Options Volume')
# axes[1, 1].set_ylabel('Volume')
axes[1, 1].set_ylabel('Volume')
# 

# Plot 6: Options Pricing and Volume - 2
# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# plt.tight_layout()
plt.tight_layout()
# 

# Assuming op_data is the list of dates you provided
# op_data = options_data("SPY",last_december_index)
op_data = options_data("SPY",last_december_index)
# 

# option_chain = op_data[1]
option_chain = op_data[1]
# axes[1, 2].plot(option_chain.calls.strike, option_chain.calls.lastPrice, 'ro', label='Calls Price')
axes[1, 2].plot(option_chain.calls.strike, option_chain.calls.lastPrice, 'ro', label='Calls Price')
# axes[1, 2].plot(option_chain.puts.strike, option_chain.puts.lastPrice, 'bo', label='Puts Price')
axes[1, 2].plot(option_chain.puts.strike, option_chain.puts.lastPrice, 'bo', label='Puts Price')
# axes[1, 2].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
axes[1, 2].axvline(x= op_data[2].values  , color='k', linestyle='--', linewidth=1)
# axes[1, 2].legend(loc='upper right')
axes[1, 2].legend(loc='upper right')
# axes[1, 2].set_xlabel('Strike Price')
axes[1, 2].set_xlabel('Strike Price')
# axes[1, 2].set_ylabel('$')
axes[1, 2].set_ylabel('$')
# 

# ax2 = axes[1, 2].twinx()
ax2 = axes[1, 2].twinx()
# ax2.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility,'go',label='Calls IV')
ax2.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility,'go',label='Calls IV')
# ax2.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility,'mo',label='Puts IV')
ax2.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility,'mo',label='Puts IV')
# ax2.legend(loc='lower right')
ax2.legend(loc='lower right')
# ax2.set_ylabel('Volume')
ax2.set_ylabel('Volume')
# 

# plt.tight_layout()
plt.tight_layout()
# plt.savefig('spy_plots.png')
plt.savefig('spy_plots.png')
# plt.show()
plt.show()
# 

# 

# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# plt.figure(figsize=(18,8))
plt.figure(figsize=(18,8))
# plt.tight_layout()
plt.tight_layout()
# plt.title('Equity Sectors Trailling Correlation')
plt.title('Equity Sectors Trailling Correlation')
# plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(12).corr()['^GSPC']*100,color='none', edgecolor='blue', linewidth=2)
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(12).corr()['^GSPC']*100,color='none', edgecolor='blue', linewidth=2)
# plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(6).corr()['^GSPC']*100,color='none', edgecolor='red', linewidth=2)
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(6).corr()['^GSPC']*100,color='none', edgecolor='red', linewidth=2)
# plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(3).corr()['^GSPC']*100,color='none', edgecolor='green', linewidth=2)
plt.bar(sectors_names,sectors_data.groupby(pd.Grouper(freq='1ME')).mean().tail(3).corr()['^GSPC']*100,color='none', edgecolor='green', linewidth=2)
# plt.grid(True)
plt.grid(True)
# plt.legend(['12 Months','6 Months','3 Months'])
plt.legend(['12 Months','6 Months','3 Months'])
# plt.savefig("sectors correlation.png")
plt.savefig("sectors correlation.png")
# 

# plt.figure(figsize=(18,8))
plt.figure(figsize=(18,8))
# 

# Plot the bars
# plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-2])-1)*100 , color='none', edgecolor='blue', linewidth=2, label='1-day Growth')
plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-2])-1)*100 , color='none', edgecolor='blue', linewidth=2, label='1-day Growth')
# plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-8])-1)*100 , color='none', edgecolor='red', linewidth=2, label='7-day Growth')
plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-8])-1)*100 , color='none', edgecolor='red', linewidth=2, label='7-day Growth')
# plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-31])-1)*100 , color='none', edgecolor='green', linewidth=2, label='30-day Growth')
plt.bar(sectors_names, ((sectors_data.iloc[len(sectors_data)-1]/sectors_data.iloc[len(sectors_data)-31])-1)*100 , color='none', edgecolor='green', linewidth=2, label='30-day Growth')
# 

# Adding labels and title
# plt.xlabel('Sectors')
plt.xlabel('Sectors')
# plt.ylabel('Growth')
plt.ylabel('Growth')
# plt.title('Equity Sectors Perfomance')
plt.title('Equity Sectors Perfomance')
# plt.legend()  # Add a legend to distinguish between different time periods
plt.legend()  # Add a legend to distinguish between different time periods
# 

# plt.tight_layout()
plt.tight_layout()
# plt.savefig('equity sectors perfomance.png')
plt.savefig('equity sectors perfomance.png')
# 

# 

# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# plt.tight_layout()
plt.tight_layout()
# 

# tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLC', 'XLU']
tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLC', 'XLU']
# num_plots = len(tickers)
num_plots = len(tickers)
# sectors_names = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Materials', 'Real Estate', 'Technology', 'Communication Services', 'Utilities']
sectors_names = ['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Materials', 'Real Estate', 'Technology', 'Communication Services', 'Utilities']
# 

# num_rows = 3
num_rows = 3
# num_cols = 4
num_cols = 4
# 

# Create subplots in a 3x4 grid
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
# 

# Flatten the axes array to iterate over it in a single loop
# axes = axes.flatten()
axes = axes.flatten()
# 

# for ax, ticker, sector_name in zip(axes, tickers, sectors_names):
for ax, ticker, sector_name in zip(axes, tickers, sectors_names):
# option_chain = options_data(ticker, 1)[1]
    option_chain = options_data(ticker, 1)[1]
# ax.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility, 'ro', label=f'{ticker} Calls')
    ax.plot(option_chain.calls.strike, option_chain.calls.impliedVolatility, 'ro', label=f'{ticker} Calls')
# ax.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility, 'bo', label=f'{ticker} Puts')
    ax.plot(option_chain.puts.strike, option_chain.puts.impliedVolatility, 'bo', label=f'{ticker} Puts')
# ax.axvline(x= options_data(ticker,1)[2].values  , color='k', linestyle='--', linewidth=1)
    ax.axvline(x= options_data(ticker,1)[2].values  , color='k', linestyle='--', linewidth=1)
# ax.legend(loc='upper right')
    ax.legend(loc='upper right')
# ax.set_xlabel('Strike Price')
    ax.set_xlabel('Strike Price')
# ax.set_ylabel('Implied Volatility')
    ax.set_ylabel('Implied Volatility')
# ax.set_title(f'Options Volatility - {sector_name}')
    ax.set_title(f'Options Volatility - {sector_name}')
# 

# Adjust layout
# plt.tight_layout()
plt.tight_layout()
# 

# plt.savefig('equity sectors options volatility.png')
plt.savefig('equity sectors options volatility.png')
# 

# 

# In[19]:
# 

# 

# Yields Plots
# yields_plot =yields.tail(10)
yields_plot =yields.tail(10)
# 

# 

# Create a figure and axes for the grid
# fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# 

# Plot 1: Yield Curve
# sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("whitegrid", {'axes.grid' : False})
# axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-6], marker='o', linestyle='-',color='black')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-6], marker='o', linestyle='-',color='black')
# axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-1], marker='o', linestyle='-',color='blue')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-1], marker='o', linestyle='-',color='blue')
# axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-13], marker='o', linestyle='-',color='red')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-13], marker='o', linestyle='-',color='red')
# axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-29], marker='o', linestyle='-',color='green')
axes[0].plot(yields_plot.columns, yields.iloc[len(yields.index)-29], marker='o', linestyle='-',color='green')
# axes[0].legend([yields.iloc[len(yields.index)-6].name,
axes[0].legend([yields.iloc[len(yields.index)-6].name,
# yields.iloc[len(yields.index)-1].name,
                yields.iloc[len(yields.index)-1].name,
# yields.iloc[len(yields.index)-13].name,
                yields.iloc[len(yields.index)-13].name,
# yields.iloc[len(yields.index)-29].name])
                yields.iloc[len(yields.index)-29].name])
# axes[0].set_title('Yield Curve')
axes[0].set_title('Yield Curve')
# axes[0].set_xlabel('Maturity')
axes[0].set_xlabel('Maturity')
# axes[0].set_ylabel('Interest Rate')
axes[0].set_ylabel('Interest Rate')
# 

# Plot 2: Yield Historical
# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# axes[1].plot(yields_plot.index,yields_plot['1mo'],'-o')
axes[1].plot(yields_plot.index,yields_plot['1mo'],'-o')
# axes[1].plot(yields_plot.index,yields_plot['1yr'],'-o')
axes[1].plot(yields_plot.index,yields_plot['1yr'],'-o')
# axes[1].plot(yields_plot.index,yields_plot['10yr'],'-o')
axes[1].plot(yields_plot.index,yields_plot['10yr'],'-o')
# axes[1].plot(yields_plot.index,yields_plot['30yr'],'-o')
axes[1].plot(yields_plot.index,yields_plot['30yr'],'-o')
# axes[1].set_title('Yield Historical')
axes[1].set_title('Yield Historical')
# axes[1].legend(['1mo','1yr','10yr','30yr'])
axes[1].legend(['1mo','1yr','10yr','30yr'])
# axes[1].set_xlabel('Date')
axes[1].set_xlabel('Date')
# axes[1].set_ylabel('Interest Rate')
axes[1].set_ylabel('Interest Rate')
# 

# Adjust layout
# plt.tight_layout()
plt.tight_layout()
# 

# Save the figure
# plt.savefig('treasury curve.png')
plt.savefig('treasury curve.png')
# 

# 

# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# plt.tight_layout()
plt.tight_layout()
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
# sns.heatmap(yields_plot, annot=True, fmt=".2f", cmap="inferno")
sns.heatmap(yields_plot, annot=True, fmt=".2f", cmap="inferno")
# plt.title('Yield Curve Trailling')
plt.title('Yield Curve Trailling')
# plt.ylabel('Date')
plt.ylabel('Date')
# plt.savefig('yield curve trailling heatmap.png')
plt.savefig('yield curve trailling heatmap.png')
# 

# 

# In[20]:
# 

# 

#CPI Plots
# consumer_plot = consumer.tail(12)
consumer_plot = consumer.tail(12)
# prices_plot = prices.tail(12)
prices_plot = prices.tail(12)
# 

# Create a figure and axes for the grid
# fig, axes = plt.subplots(1, 2, figsize=(20, 6))
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# 

# Plot 1: CPI Categories Heatmap
# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# sns.heatmap(consumer_plot, annot=True, fmt=".2f", cmap="inferno", ax=axes[0])
sns.heatmap(consumer_plot, annot=True, fmt=".2f", cmap="inferno", ax=axes[0])
# axes[0].set_title('CPI Categories')
axes[0].set_title('CPI Categories')
# axes[0].set_ylabel('Date')
axes[0].set_ylabel('Date')
# 

# Plot 2: Price Indexes YoY% Change
# sns.set_style("whitegrid")
sns.set_style("whitegrid")
# axes[1].plot(prices_plot)
axes[1].plot(prices_plot)
# axes[1].set_title('Price Indexes YoY% Change')
axes[1].set_title('Price Indexes YoY% Change')
# axes[1].legend(['CPI','PCE','PPI'])
axes[1].legend(['CPI','PCE','PPI'])
# 

# Adjust layout
# plt.tight_layout()
plt.tight_layout()
# 

# Save the figure
# plt.savefig('cpi plots.png')
plt.savefig('cpi plots.png')
# 

# Show the plot
# plt.show()
plt.show()
# 

# 

# In[21]:
# 

# 

# 

# Labor Market Plots
# labor_market_plot = labor_market.tail(24)
labor_market_plot = labor_market.tail(24)
# 

# Create a figure and axes for the grid
# fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
# 

# Plot 1: Employment Indicators
# axes[0, 0].plot(labor_market_plot.index, labor_market_plot['Nonfarm Change'], color='black', label='Payems Change')
axes[0, 0].plot(labor_market_plot.index, labor_market_plot['Nonfarm Change'], color='black', label='Payems Change')
# axes[0, 0].set_xlabel('Date')
axes[0, 0].set_xlabel('Date')
# axes[0, 0].set_ylabel('Payems Change')
axes[0, 0].set_ylabel('Payems Change')
# axes[0, 0].legend(['NonFarm Payroll'])
axes[0, 0].legend(['NonFarm Payroll'])
# 

# Create a secondary y-axis
# ax2 = axes[0, 0].twinx()
ax2 = axes[0, 0].twinx()
# Plot on the secondary y-axis
# ax2.plot(labor_market_plot.index, labor_market_plot['Unemployment Rate'], color='green', label='Unemployment Rate')
ax2.plot(labor_market_plot.index, labor_market_plot['Unemployment Rate'], color='green', label='Unemployment Rate')
# ax2.set_ylabel('Unemployment Rate')
ax2.set_ylabel('Unemployment Rate')
# ax2.legend(['Unemployment Rate'])
ax2.legend(['Unemployment Rate'])
# axes[0, 0].set_title('Employment Indicators')
axes[0, 0].set_title('Employment Indicators')
# 

# Plot 2: Unemployment Rate
# axes[0, 1].grid(True)
axes[0, 1].grid(True)
# axes[0, 1].plot(labor_market_plot['Unemployment Rate'],'-ok')
axes[0, 1].plot(labor_market_plot['Unemployment Rate'],'-ok')
# axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(3).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(3).mean().tail(len(labor_market_plot)),'o')
# axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(6).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(6).mean().tail(len(labor_market_plot)),'o')
# axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(12).mean().tail(len(labor_market_plot)),'o')
axes[0, 1].plot(labor_market['Unemployment Rate'].rolling(12).mean().tail(len(labor_market_plot)),'o')
# axes[0, 1].set_title('Unemployment Rate')
axes[0, 1].set_title('Unemployment Rate')
# axes[0, 1].set_ylabel('%')
axes[0, 1].set_ylabel('%')
# axes[0, 1].set_xlabel('Date')
axes[0, 1].set_xlabel('Date')
# axes[0, 1].legend(['Current Rate','3M MA','6M MA','12M MA'])
axes[0, 1].legend(['Current Rate','3M MA','6M MA','12M MA'])
# 

# Plot 3: Labor Forces
# axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Demand'],'-o')
axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Demand'],'-o')
# axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Supply'],'-o')
axes[1, 0].plot(labor_market_plot.index, labor_market_plot['Labor Supply'],'-o')
# axes[1, 0].set_ylabel('Thousands of Persons')
axes[1, 0].set_ylabel('Thousands of Persons')
# ax2 = axes[1, 0].twinx()
ax2 = axes[1, 0].twinx()
# ax2.plot(labor_market_plot.index, labor_market_plot['Labor Difference'],'k-o')
ax2.plot(labor_market_plot.index, labor_market_plot['Labor Difference'],'k-o')
# ax2.set_ylabel('Labor Difference')
ax2.set_ylabel('Labor Difference')
# axes[1, 0].set_title('Labor Forces')
axes[1, 0].set_title('Labor Forces')
# axes[1, 0].legend(['Labor Demand','Labor Supply'])
axes[1, 0].legend(['Labor Demand','Labor Supply'])
# 

# Plot 4: Weekly Earnings
# axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Weekly Hours'],'b-o')
axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Weekly Hours'],'b-o')
# axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Hourly Earnings'],'g-o')
axes[1, 1].plot(labor_market_plot.index, labor_market_plot['Average Hourly Earnings'],'g-o')
# axes[1, 1].legend(['Average Weekly Hours','Average Hourly Earnings'])
axes[1, 1].legend(['Average Weekly Hours','Average Hourly Earnings'])
# ax2 = axes[1, 1].twinx()
ax2 = axes[1, 1].twinx()
# ax2.plot(labor_market_plot.index, labor_market_plot['AWH YoY%'],'-o')
ax2.plot(labor_market_plot.index, labor_market_plot['AWH YoY%'],'-o')
# ax2.plot(labor_market_plot.index, labor_market_plot['AHE YoY%'],'-o')
ax2.plot(labor_market_plot.index, labor_market_plot['AHE YoY%'],'-o')
# axes[1, 1].legend(['Average Weekly Hours YoY%','Average Hourly Earnings YoY%'])
axes[1, 1].legend(['Average Weekly Hours YoY%','Average Hourly Earnings YoY%'])
# axes[1, 1].set_title('Weekly Earnings')
axes[1, 1].set_title('Weekly Earnings')
# 

# Adjust layout
# plt.tight_layout()
plt.tight_layout()
# 

# Save the figure
# plt.savefig('labor indicators.png')
plt.savefig('labor indicators.png')
# 

# Show the plot
# plt.show()
plt.show()
# 

# 

# Converting data and plots into a PDF
# 

# In[22]:
# 

# 

# today =datetime.now().strftime("%Y-%m-%d @ %H:%M")
today =datetime.now().strftime("%Y-%m-%d @ %H:%M")
# date_time = datetime.now().strftime("%Y-%m-%d")
date_time = datetime.now().strftime("%Y-%m-%d")
# 

# context = {'date_time':today,
context = {'date_time':today,
# 'sp500_value':  "${:,.2f}".format(summary[date_time].iloc[0])
           'sp500_value':  "${:,.2f}".format(summary[date_time].iloc[0])
# ,'sp500_daily': "{:,.2f}%".format(summary[date_time].iloc[1])
           ,'sp500_daily': "{:,.2f}%".format(summary[date_time].iloc[1])
# ,'sp500_30d_std': "{:,.2f}".format(summary[date_time].iloc[2])
           ,'sp500_30d_std': "{:,.2f}".format(summary[date_time].iloc[2])
# ,'vix': "{:,.2f}".format(summary[date_time].iloc[3])
           ,'vix': "{:,.2f}".format(summary[date_time].iloc[3])
# ,'tbill': "{:,.2f}".format(summary[date_time].iloc[4])
           ,'tbill': "{:,.2f}".format(summary[date_time].iloc[4])
# ,'tbond': "{:,.2f}".format(summary[date_time].iloc[5])
           ,'tbond': "{:,.2f}".format(summary[date_time].iloc[5])
# ,'yields_table' : yields_plot.tail(5).to_html()
           ,'yields_table' : yields_plot.tail(5).to_html()
# }
           }
# 

# template_loader = jinja2.FileSystemLoader('./')
template_loader = jinja2.FileSystemLoader('./')
# template_env = jinja2.Environment(loader=template_loader)
template_env = jinja2.Environment(loader=template_loader)
# 

# template = template_env.get_template("Econ Report Template.html")
template = template_env.get_template("Econ Report Template.html")
# output_text = template.render(context)
output_text = template.render(context)
# 

# file_path = "Econ Report.html"
file_path = "Econ Report.html"
# with open(file_path, "w",encoding="utf-8") as file:
with open(file_path, "w",encoding="utf-8") as file:
# file.write(output_text)
    file.write(output_text)
# 

# 

# In[23]:
# 

# 

# yields.index = pd.to_datetime(yields.index)
yields.index = pd.to_datetime(yields.index)
# yields_annual = yields.groupby(pd.Grouper(freq='1YE')).mean()
yields_annual = yields.groupby(pd.Grouper(freq='1YE')).mean()
# 

# 

# In[24]:
# 

# 

# prod.index = pd.to_datetime(prod.index)
prod.index = pd.to_datetime(prod.index)
# prod_annual = prod.groupby(pd.Grouper(freq='1YE')).sum()
prod_annual = prod.groupby(pd.Grouper(freq='1YE')).sum()
# prod_annual.index = pd.to_datetime(prod_annual.index)
prod_annual.index = pd.to_datetime(prod_annual.index)
# 

# 

# prod_annual['GDP YoY'] = prod_annual['GDP'].pct_change()
prod_annual['GDP YoY'] = prod_annual['GDP'].pct_change()
# prod_annual['RGDP YoY'] = prod_annual['GDPC1'].pct_change()
prod_annual['RGDP YoY'] = prod_annual['GDPC1'].pct_change()
# prod_annual['GDI YoY'] = prod_annual['GDI'].pct_change()
prod_annual['GDI YoY'] = prod_annual['GDI'].pct_change()
# prod_annual['RGDI YoY'] = prod_annual['A261RX1Q020SBEA'].pct_change()
prod_annual['RGDI YoY'] = prod_annual['A261RX1Q020SBEA'].pct_change()
# 

# prod['GDP QoQ'] = prod['GDP'].pct_change()
prod['GDP QoQ'] = prod['GDP'].pct_change()
# prod['RGDP QoQ'] = prod['GDPC1'].pct_change()
prod['RGDP QoQ'] = prod['GDPC1'].pct_change()
# prod['GDI QoQ'] = prod['GDI'].pct_change()
prod['GDI QoQ'] = prod['GDI'].pct_change()
# prod['RGDI QoQ'] = prod['A261RX1Q020SBEA'].pct_change()
prod['RGDI QoQ'] = prod['A261RX1Q020SBEA'].pct_change()
# 

# prod['GDP Growth Annualized'] = (1 + prod['GDP QoQ'])**4 - 1
prod['GDP Growth Annualized'] = (1 + prod['GDP QoQ'])**4 - 1
# prod['RGDP Growth Annualized'] = (1 + prod['RGDP QoQ'])**4 - 1
prod['RGDP Growth Annualized'] = (1 + prod['RGDP QoQ'])**4 - 1
# prod['GDI Growth Annualized'] = (1 + prod['GDI QoQ'])**4 - 1
prod['GDI Growth Annualized'] = (1 + prod['GDI QoQ'])**4 - 1
# prod['RGDI Growth Annualized'] = (1 + prod['RGDI QoQ'])**4 - 1
prod['RGDI Growth Annualized'] = (1 + prod['RGDI QoQ'])**4 - 1
# 

# prod.replace('inf',0)
prod.replace('inf',0)
# 

# 

# In[25]:
# 

# 

# sp_annual = pd.DataFrame(sp.close.groupby(pd.Grouper(freq='1YE')).last())
sp_annual = pd.DataFrame(sp.close.groupby(pd.Grouper(freq='1YE')).last())
# sp_annual['yoy'] = sp_annual.close.pct_change()
sp_annual['yoy'] = sp_annual.close.pct_change()
# sp_annual
sp_annual
# 

# 

# Creating Excel File
# 

# In[26]:
# 

# 

# with pd.ExcelWriter('Econ Data Link.xlsx', engine='xlsxwriter') as writer:
with pd.ExcelWriter('Econ Data Link.xlsx', engine='xlsxwriter') as writer:
# yields.to_excel(writer, sheet_name='Daily Yields', index=True)
    yields.to_excel(writer, sheet_name='Daily Yields', index=True)
# yields_annual.to_excel(writer, sheet_name='Annualized Yields', index=True)
    yields_annual.to_excel(writer, sheet_name='Annualized Yields', index=True)
# prod.to_excel(writer,sheet_name='National Product',index=True)
    prod.to_excel(writer,sheet_name='National Product',index=True)
# prod_annual.to_excel(writer,sheet_name='Annualized National Product',index=True)
    prod_annual.to_excel(writer,sheet_name='Annualized National Product',index=True)
# sp.to_excel(writer, sheet_name='S&P 500',index=True)
    sp.to_excel(writer, sheet_name='S&P 500',index=True)
# sp_annual.to_excel(writer, sheet_name='Annualized S&P 500',index=True)
    sp_annual.to_excel(writer, sheet_name='Annualized S&P 500',index=True)
# 

