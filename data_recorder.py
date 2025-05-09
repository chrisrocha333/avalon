import pandas as pd
import numpy as np
import yfinance as yf
from full_fred.fred import Fred
from datetime import date
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

print("Starting data collection process...")

key = 'G:/My Drive/Markets Analysis/Data Collection/key.txt'

#Set API key
try:
    print("Attempting to set up FRED API connection...")
    fred = Fred(key)
    fred.set_api_key_file(key)
    print("Successfully connected to FRED API")
except:
    print('Error: API key not found')

#functions
def daily(x):
    print(f"Downloading daily data for {x}...")
    dta = yf.download(tickers=x,period='max',interval = '1d')
    dta[x] = dta['Close']
    dta = dta.drop(columns=['Open','High','Low','Close','Volume'])
    dta[x] = np.array(dta[x])
    print(f"Successfully downloaded daily data for {x}")
    return dta[x]

def fed_annual(x):
    print(f"Fetching annual FRED data for {x}...")
    a = fred.get_series_df(x,frequency='a')
    a.value = a.value.replace('.',np.nan)
    a.value = a.value.ffill()
    a.index = pd.to_datetime(a.date)
    a = a.drop(columns=['date','realtime_start','realtime_end'])
    a.value = a.value.astype('float')
    print(f"Successfully processed annual data for {x}")
    return a

def data(x,y):
    print(f"Fetching {y} frequency FRED data for {x}...")
    a = fred.get_series_df(x,frequency=y)
    a.value = a.value.replace('.',np.nan)
    a.value = a.value.ffill()
    a.index = a.date
    a = a.drop(columns=['date','realtime_start','realtime_end'])
    a.value = a.value.astype('float')
    print(f"Successfully processed {y} frequency data for {x}")
    return a

print("\nProcessing FRED data IDs...")
#Fed Data id codes
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

print(f"Total number of FRED IDs to process: {len(ids)}")

#Labor Market
print("\nSetting up data categories...")
labor_market = list(['PAYEMS','CIVPART','UNRATE','JTSJOL','ICSA','AWHAETP','CES0500000003','CE16OV','CLF16OV'])
print(f"Labor market indicators: {len(labor_market)}")

#Rates
rates = list(['DTB4WK','DTB3','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30','DFF','DCPF3M'])
print(f"Rate indicators: {len(rates)}")

#Production
production = list(['GDP','GDPC1','GDI','A261RX1Q020SBEA'])
print(f"Production indicators: {len(production)}")

#Consumer Spending
consumer_spending = list(['TOTALSA','RETAILIMSA','TOTBUSSMSA','BOPGSTB','RSCCAS','RSXFS','BUSINV','ATLSBUSRGEP','DSPIC96'])
print(f"Consumer spending indicators: {len(consumer_spending)}")

#Consumer Debt
consumer_debt = list(['TOTALSL','GFDEBTN','H8B1058NCBCMG','RESPPANWW','TMBACBW027SBOG','DRALACBS'])
print(f"Consumer debt indicators: {len(consumer_debt)}")

#Housing
housing = list(['RHORUSQ156N','CSUSHPINSA','TTLCONS','ACTLISCOUUS','EXHOSLUSM495S','HOSMEDUSM052N'])
print(f"Housing indicators: {len(housing)}")

#Monetary & Prices
prices = list(['CPIAUCSL','PCE','PPIACO'])
money = list(['M2','M2V'])
print(f"Price indicators: {len(prices)}")
print(f"Money indicators: {len(money)}")

#Government Debt
government_debt = list(['QBPBSTAS','QBPBSTASSCUSTRSC','WSHOMCB','GFDEBTN'])
print(f"Government debt indicators: {len(government_debt)}")

#Trade
trade = list(['FRGSHPUSM649NCIS','BOPGSTB','PETROLEUMD11','DTCDISA066MSFRBNY'])
print(f"Trade indicators: {len(trade)}")

print("\nProcessing FRED metadata...")
df = pd.DataFrame()

for i in ids:
    print(f"Processing metadata for {i}...")
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
print("Metadata processing complete")

print("\nProcessing main data series...")
ids = list(df.id)
names = list(df.title)

print("Processing annual data...")
dta = pd.DataFrame()
for i in ids:
    print(f"Processing annual data for {i}...")
    y = fed_annual(i)
    dta[i] = y.value

print("\nProcessing production data...")
prod = pd.DataFrame()
for i in production:
    print(f"Processing production data for {i}...")
    y = data(i,'q')
    prod[i] = y.value

print("\nProcessing labor market data...")
lm = pd.DataFrame()
for i in labor_market:
    print(f"Processing labor market data for {i}...")
    y = data(i,'m')
    lm[i] = y.value

print("Calculating labor market metrics...")
lm['labor_demand'] = (lm['CLF16OV'] + lm.JTSJOL)/1000
lm['labor_supply'] = lm['CE16OV']/1000
lm['labor_difference'] = lm['labor_supply']-lm['labor_demand']
lm['payems_change'] = lm['PAYEMS']-lm['PAYEMS'].shift(1)
lm['awh_yoy'] = (lm['AWHAETP']/lm['AWHAETP'].shift(12)-1)*100
lm['ahe_yoy'] = (lm['CES0500000003']/lm['CES0500000003'].shift(12)-1)*100
lm.columns = ["All Employees, Total Nonfarm", "Labor Force Participation Rate", "Unemployment Rate", "Job Openings: Total Nonfarm","Initial Claims", "Average Weekly Hours", "Average Hourly Earnings","Civilial Labor Force Level","Employment Level","Labor Demand","Labor Supply","Labor Difference","Nonfarm Change","AWH YoY%","AHE YoY%"]
print("Labor market processing complete")

print("\nProcessing rates data...")
r = pd.DataFrame()
for i in rates:
    print(f"Processing rates data for {i}...")
    y = data(i,'d')
    r[i] = y.value
r.index = pd.to_datetime(r.index)

print("\nProcessing yields data...")
yields = r.drop(columns=['DFF','DCPF3M'])
yields.columns = ['1mo','3mo','1yr','2yr','3yr','5yr','7yr','10yr','20yr','30yr']
yield_curve = yields.loc[r.index.max()]
yields.index = yields.index.strftime('%Y-%m-%d')
print("Rates and yields processing complete")

print("\nProcessing corporate debt data...")
corp = list(['BAMLC1A0C13YEY','BAMLC2A0C35YEY','BAMLC3A0C57YEY','BAMLC4A0C710YEY','BAMLC7A0C1015YEY','BAMLC8A0C15PYEY','BAMLH0A0HYM2EY','BAMLC0A0CMEY'])
corp_debt = pd.DataFrame()
for i in corp:
    print(f"Processing corporate debt data for {i}...")
    corp_debt[i] = data(i,'d')

corp_debt.index = pd.to_datetime(corp_debt.index)
corp_debt_y = corp_debt.groupby(pd.Grouper(freq='1YE')).mean()
corp_debt_std = corp_debt.rolling(21).std()
corp = corp_debt
corp.columns = ['1-3yrs','3-5yrs','5-7yrs','7-10yrs','10-15yrs','15+ yrs','High Yield Index','US Corp Index']
corp_curve = corp.drop(columns=['High Yield Index','US Corp Index'])
corp_curve = corp_curve.loc[corp_curve.index.max()]
print("Corporate debt processing complete")

print("\nProcessing CPI indicators...")
indicators = list(['CPIUFDSL','CUSR0000SAF11','CUSR0000SEFV','CPIENGSL','CUSR0000SETB01','CUSR0000SEHE','CUSR0000SEHF','CUSR0000SEHF01','CUSR0000SACL1E','CUSR0000SAD','CUSR0000SAN','CUSR0000SASLE','CUSR0000SAH1','CUSR0000SAM2','CUSR0000SAS4','CPIAUCSL'])
cpi = pd.DataFrame()
for i in indicators:
    print(f"Processing CPI indicator {i}...")
    cpi[i] = data(i,'m')

cpi_yoy = (cpi / cpi.shift(12))-1
print("CPI processing complete")

print("\nProcessing price indices...")
p = pd.DataFrame()
for i in prices:
    print(f"Processing price index {i}...")
    y = data(i,'m')
    p[i] = y.value

p['cpi_growth_yoy'] = p.CPIAUCSL / p.CPIAUCSL.shift(12) - 1
p['pce_growth_yoy'] = p.PCE / p.PCE.shift(12) - 1
p['ppi_growth_yoy'] = p.PPIACO / p.PPIACO.shift(12) - 1
print("Price indices processing complete")

print("\nProcessing CPI components...")
consumer = cpi_yoy*100
consumer.columns = ["All items","Food","Food at home","Food away from home(1)","Energy","Gasoline","Fuel Oil","Energy services","Electricity","Commodities","Durable Goods","Non-Durable Goods","Service","Shelter","Medical","Transportation"]
print("CPI components processing complete")

print("\nProcessing price indices...")
prices = p*100
prices.index = pd.to_datetime(prices.index)
prices = prices.drop(columns=['CPIAUCSL','PCE','PPIACO'])
prices.columns = ['CPI YoY','PCE YoY','PPI YoY']
print("Price indices processing complete")

print("\nProcessing S&P 500 data...")
sp = pd.DataFrame()
sp['close'] = daily('^GSPC')
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
print("S&P 500 processing complete")

print("\nCreating summary data...")
summary_rows = ['S&P 500','S&P 500 Daily Growth','30d Moving Sigma','VIX','1mo Bill','10yr Treasury']
summary = pd.DataFrame(index=summary_rows)

today_date = date.today()
today_date = today_date.strftime("%Y-%m-%d")
print(f"Today's date: {today_date}")

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
print("Summary data creation complete")

print("\nProcessing market sectors data...")
sectors = ['XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLC','XLU','^GSPC']
sectors_names = ['Cons. Discreat.','Cons. Staples','Energy','Financials',
                 'Health Care','Industrials','Materials','Real Estate','Tech',
                 'Comms','Utilitites','S&P 500']

sectors_data = daily(sectors)
sectors_data.index = pd.to_datetime(sectors_data.index)
print("Market sectors processing complete")

# Export data to Excel
def export_to_excel():
    print("\nStarting Excel export process...")
    # Create Excel writer object
    excel_file = 'Data Collection/market_data.xlsx'
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

    print("Exporting dataframes to Excel sheets...")
    # Export each dataframe to a separate sheet
    lm.to_excel(writer, sheet_name='Labor Market')
    yields.to_excel(writer, sheet_name='Treasury Yields')
    corp_debt.to_excel(writer, sheet_name='Corporate Debt')
    consumer.to_excel(writer, sheet_name='CPI Components')
    prices.to_excel(writer, sheet_name='Price Indices')
    sp.to_excel(writer, sheet_name='S&P 500')
    sectors_data.to_excel(writer, sheet_name='Market Sectors')

    print("Setting up Excel formatting...")
    # Get the workbook and create a format for dates
    workbook = writer.book
    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})

    # Format each sheet
    for sheet_name in writer.sheets:
        print(f"Formatting sheet: {sheet_name}")
        worksheet = writer.sheets[sheet_name]
        
        # Set column widths
        worksheet.set_column('A:A', 15)  # Date column
        worksheet.set_column('B:Z', 12)  # Data columns
        
        # Format date column
        worksheet.set_column('A:A', 15, date_format)
        
        # Add filters to all columns
        worksheet.autofilter(0, 0, worksheet.dim_rowmax, worksheet.dim_colmax)

    # Save the Excel file
    writer.close()
    print(f"Excel file '{excel_file}' has been created successfully!")

def export_to_mssql():
    print("\nStarting MSSQL export process...")
    
    # Connection parameters - you'll need to update these with your actual database details
    server ='localhost'
    port = '1433'
    database = 'master'
    username = 'sa'
    password = 'yourStrong(!)Password'
    
    # Create SQLAlchemy engine
    conn_str = f'mssql+pyodbc://{username}:{password}@{server}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server'
    
    try:
        print("Connecting to MSSQL database...")
        engine = create_engine(conn_str)
        connection = engine.connect()
        print("Successfully connected to database")
        
        # Create tables if they don't exist
        print("Creating tables if they don't exist...")
        
        # Create tables with proper data types
        create_tables_sql = """
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'LaborMarket')
        CREATE TABLE LaborMarket (
            date DATETIME,
            [All Employees, Total Nonfarm] FLOAT,
            [Labor Force Participation Rate] FLOAT,
            [Unemployment Rate] FLOAT,
            [Job Openings: Total Nonfarm] FLOAT,
            [Initial Claims] FLOAT,
            [Average Weekly Hours] FLOAT,
            [Average Hourly Earnings] FLOAT,
            [Civilial Labor Force Level] FLOAT,
            [Employment Level] FLOAT,
            [Labor Demand] FLOAT,
            [Labor Supply] FLOAT,
            [Labor Difference] FLOAT,
            [Nonfarm Change] FLOAT,
            [AWH YoY%] FLOAT,
            [AHE YoY%] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'TreasuryYields')
        CREATE TABLE TreasuryYields (
            date DATETIME,
            [1mo] FLOAT,
            [3mo] FLOAT,
            [1yr] FLOAT,
            [2yr] FLOAT,
            [3yr] FLOAT,
            [5yr] FLOAT,
            [7yr] FLOAT,
            [10yr] FLOAT,
            [20yr] FLOAT,
            [30yr] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'CorporateDebt')
        CREATE TABLE CorporateDebt (
            date DATETIME,
            [1-3yrs] FLOAT,
            [3-5yrs] FLOAT,
            [5-7yrs] FLOAT,
            [7-10yrs] FLOAT,
            [10-15yrs] FLOAT,
            [15+ yrs] FLOAT,
            [High Yield Index] FLOAT,
            [US Corp Index] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'CPIComponents')
        CREATE TABLE CPIComponents (
            date DATETIME,
            [All items] FLOAT,
            [Food] FLOAT,
            [Food at home] FLOAT,
            [Food away from home(1)] FLOAT,
            [Energy] FLOAT,
            [Gasoline] FLOAT,
            [Fuel Oil] FLOAT,
            [Energy services] FLOAT,
            [Electricity] FLOAT,
            [Commodities] FLOAT,
            [Durable Goods] FLOAT,
            [Non-Durable Goods] FLOAT,
            [Service] FLOAT,
            [Shelter] FLOAT,
            [Medical] FLOAT,
            [Transportation] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'PriceIndices')
        CREATE TABLE PriceIndices (
            date DATETIME,
            [CPI YoY] FLOAT,
            [PCE YoY] FLOAT,
            [PPI YoY] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'SP500')
        CREATE TABLE SP500 (
            date DATETIME,
            [close] FLOAT,
            [100ma] FLOAT,
            [200ma] FLOAT,
            [30mstd] FLOAT,
            [60mstd] FLOAT,
            [100mstd] FLOAT,
            [200mstd] FLOAT,
            [zscore 100ma] FLOAT,
            [daily] FLOAT,
            [mom] FLOAT,
            [yoy] FLOAT,
            [ytd] FLOAT,
            [vix] FLOAT
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'MarketSectors')
        CREATE TABLE MarketSectors (
            date DATETIME,
            [Cons. Discreat.] FLOAT,
            [Cons. Staples] FLOAT,
            [Energy] FLOAT,
            [Financials] FLOAT,
            [Health Care] FLOAT,
            [Industrials] FLOAT,
            [Materials] FLOAT,
            [Real Estate] FLOAT,
            [Tech] FLOAT,
            [Comms] FLOAT,
            [Utilitites] FLOAT,
            [S&P 500] FLOAT
        );
        """
        
        # Execute table creation
        connection.execute(text(create_tables_sql))
        connection.commit()
        
        # Insert data without creating indexes
        print("Inserting data...")
        
        # Rename columns for MarketSectors to match table definition
        sectors_data.columns = ['Cons. Discreat.', 'Cons. Staples', 'Energy', 'Financials',
                              'Health Care', 'Industrials', 'Materials', 'Real Estate', 'Tech',
                              'Comms', 'Utilitites', 'S&P 500']
        
        # Insert data
        lm.to_sql('LaborMarket', engine, if_exists='append', index=False)
        yields.to_sql('TreasuryYields', engine, if_exists='append', index=False)
        corp.to_sql('CorporateDebt', engine, if_exists='append', index=False)
        consumer.to_sql('CPIComponents', engine, if_exists='append', index=False)
        prices.to_sql('PriceIndices', engine, if_exists='append', index=False)
        sp.to_sql('SP500', engine, if_exists='append', index=False)
        sectors_data.to_sql('MarketSectors', engine, if_exists='append', index=False)
        
        # Commit the transaction
        connection.commit()
        print("Data successfully exported to MSSQL database")
        
    except SQLAlchemyError as e:
        print(f"Error exporting to MSSQL: {str(e)}")
        connection.rollback()
    
    finally:
        if 'connection' in locals():
            connection.close()
            print("Database connection closed")

if __name__ == "__main__":
    print("\nStarting main execution...")
    #export_to_excel()
    export_to_mssql()
    print("\nExecution complete!") 