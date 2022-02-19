import pandas as pd
from numpy import sqrt, log, exp, floor, busday_count, vectorize
import scipy.stats as sp
import datetime as dt
import config
import requests
from IPython.display import display
from functools import reduce, lru_cache
from utility import logg, nearest_to, deltas, retry
import json


@lru_cache(maxsize=None)
def earnings_date(stock):
    r= requests.get(f"https://finance.yahoo.com/quote/{stock.replace('.','-')}").text
    i1=0
    i1=r.find('root.App.main', i1)
    i1=r.find('{', i1)
    i2=r.find("\n", i1)
    i2=r.rfind(';', i1, i2)
    jsonstr=r[i1:i2]      
    data = json.loads(jsonstr)
    try:
        return data['context']['dispatcher']['stores']['QuoteSummaryStore']['calendarEvents']['earnings']['earningsDate'][0]['fmt']
    except:
        return '2099-12-31'

@lru_cache(maxsize=None)
def is_safe_to_write_put(stock, target_date):
    ed = earnings_date(stock)
    return trading_days(ed) < 0 or (trading_days(ed) > trading_days(target_date))  

@retry(5)
@lru_cache(maxsize=None)
@logg
def quote(stock):
    return requests.get(url=f'https://api.tdameritrade.com/v1/marketdata/{stock}/quotes?', params={'apikey': config.client_id}).json()[stock]

@retry(5)
@lru_cache(maxsize=None)
@logg
def chains(stock):
    """
                 {  "CALL" : [
                                {
                                    "expiry" : '2021-08-06'
                                    , "strikes" : {
                                                    700.0 : 5.678
                                                    , ....
                                                    }
                                }
                                , ....
                            ]
                ,"PUT" : [
                                {
                                    "expiry" : '2021-08-06'
                                    , "strikes" : {
                                                    700.0 : 5.678
                                                    , ....
                                                    }
                                }
                                , ....
                            ]
                }
    """
    d = requests.get(url=f'https://api.tdameritrade.com/v1/marketdata/chains',
                     params={'apikey': config.client_id
                         , 'symbol': stock
                             # , 'contractType': 'ALL'
                             }).json()
    return {t:
                [{
                    "expiry": e[:10]
                    , "strikes": { float(k): (lambda _: sqrt(_['bid'] * _['ask']))(v[0]) for k, v in ev.items()}}
         for e, ev in d[f'{t.lower()}ExpDateMap'].items()  if dt.datetime.fromisoformat(e[:10]) > dt.datetime.today() ] for t in ['PUT', 'CALL']}

@retry(5)
@lru_cache(maxsize=None)
@logg
def history(stock):
    return requests.get(
    f"https://api.tdameritrade.com/v1/marketdata/{stock}/pricehistory"
    , params={
        'apikey': config.client_id
        , 'periodType': 'year'
        , 'period': 1
        , 'frequencyType': 'daily'
        , 'frequency': 1
    }).json()['candles']

def option_exists(stock, target_date):
    return target_date in [_['expiry'] for _ in chains(stock)['PUT']]

def stock_prices(stock, lookback_window=250):
    extremes = lambda extremity: lambda stock: [_[extremity] for _ in history(stock)]

    alias = {'lastPrice': 'current', 'lowPrice': 'last_min', 'highPrice': 'last_max', '52WkHigh': '52weekHigh',
             '52WkLow': '52weekLow'}

    out = {alias[k]: v for k, v in quote(stock).items() if k in alias}

    for main, alt in {'last_min': 'low', 'last_max': 'high'}.items():
        if out[main] == 0:
            out[main] = extremes(alt)(stock)[-1]
    out["rSigma"] = {e: sp.cauchy.fit(list(deltas(list(map(log, extremes(e)(stock)[::-1])))))
                                  for e in ["low", "high"]}
    return out

def predicted_extreme(extremity):
    def _predicted_extreme(stock, days=1):
        return stock_prices(stock)[ 'last_min' if extremity=='min' else 'last_max' ] * exp( stock_prices(stock)['rSigma']['low' if extremity=='min' else 'high'][0] * days )
    return _predicted_extreme

def tail_probability(direction):
    def _tail_probability(stock, price, days=1):
        if direction == 'higher':
            r, sigma = stock_prices(stock)["rSigma"]["high"]
            return 1 - sp.cauchy.cdf((- r * days + log(price / stock_prices(stock)["last_max"])) / (sigma*days))
        else:
            r, sigma = stock_prices(stock)["rSigma"]["low"]
            return sp.cauchy.cdf((-r*days + log(price / stock_prices(stock)["last_min"])) / (sigma*days))
    return _tail_probability

def view_portfolio(portfolio_file
                   , sortby="gain"
                   , ascending=False
                   , columns=["stock","concentration", "gain","current"]):
    df = pd.read_csv(portfolio_file)
    df["investment"] = df["cost"]*df["quantity"]
    df["concentration"] = df["investment"] / df["investment"].sum()
    df['current'] = vectorize( lambda _ : stock_prices(_)['current'] )( df['stock'] )
    df['gain'] = df['current']/df['cost'] -1
    
    print("\nTotal investment = " + "${:,.0f}".format( df["investment"].sum() ) )
    print("\nTotal Gain = " + "{:.1%}".format( ( df['gain']*df['concentration'] ).sum() ))
    return df[columns].sort_values(sortby, ascending=ascending).style.format({
            "current": "${:,.2f}"
            , "cost": "${:,.2f}"
            , "investment": "${:,.0f}"
            , "gain": "{:.0%}"
            , "concentration": "{:.0%}"
        }).hide_index()

@lru_cache(maxsize=None)
def strike(stock, days_to_expiry, quality, distance_52weekhigh= 0.3 ):
    #df['current'] = vectorize(lambda _: stock_prices(_)['current'])(df['stock'])
    #max_strike_allowed = StockPrices.get(stock)['52weekHigh']/(1+distance_52weekhigh)
    #r, sigma = StockPrices.get(stock)["rSigma"]["low"]
    #candidate_strike = StockPrices.get(stock)['last_min'] * exp((r - (quality / 2) * sigma) * days_to_expiry)
    max_strike_allowed = stock_prices(stock)['52weekHigh']/(1+distance_52weekhigh)
    r, sigma = stock_prices(stock)["rSigma"]["low"]
    candidate_strike = stock_prices(stock)['last_min'] * exp((r - (quality / 2) * sigma) * days_to_expiry)
    return min(candidate_strike, max_strike_allowed)


@logg
@lru_cache(maxsize=None)
def option_price(option_type, stock, expiry, strike):
    return [ _ for _ in chains(stock)[option_type] if _["expiry"] == expiry ][0]["strikes"][strike]

def trading_days(t):
    return busday_count(
        (dt.datetime.now() + dt.timedelta(days= (1 if dt.datetime.now().hour > 8 else 0) )).date()
        ,  (dt.datetime.strptime(t,'%Y-%m-%d')+ dt.timedelta(days=1)).strftime('%Y-%m-%d')
    )

def get_capital(savings
                , brokerage_cash
                , adjustments
                , put_obligation_file
                , portfolio_file
                , target_date
                , risk_apetite= 1):
    thresholder = lambda c,  x : x if x >= c else 0
    #predicted_extreme(stock, extremity, days=1)
    brokerage_asset = ( lambda d : ( vectorize(min)(vectorize(thresholder)(  d['cost']
                                                        ,vectorize( predicted_extreme('max') )( d['stock']
                                                                                                , trading_days(target_date) )
                                                       ), d['cost'] )
                                *d["quantity"]).sum() 
                  )( pd.read_csv(portfolio_file) )
    
    funds = savings + brokerage_cash + adjustments + brokerage_asset
    df = pd.read_csv(put_obligation_file)
    df['price'] = df.apply(
        lambda row : sum(float(_) for _ in row['price'].split('+')) , axis=1)
    df['ProbITM'] = df.apply(
        lambda row : tail_probability('lower')(
            row['stock']
            , row['strike']
            , trading_days( row['expiry'] if row['expiry']==row['expiry'] else target_date )
        ) , axis=1)

    #df['current'] = df.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    df['current'] = vectorize(lambda _: stock_prices(_)['current'])(df['stock'])
    df['change_from_strike'] = df['current'] /df['strike'] -1
    df['earnings_date'] = vectorize(earnings_date)( df['stock'] )
    max_exposure = (df['strike']*df['num_puts']).sum()*100
    expected_exposure = (df['strike']*df['num_puts']*df['ProbITM']).sum()*100
    risk_aware_exposure= (1-risk_apetite)*max_exposure + risk_apetite*expected_exposure
    capital = funds - risk_aware_exposure
    print('\n')
    display(df.sort_values('ProbITM', ascending=False).style.format(
        {
            "strike": "${:,.1f}"
            , "price": "${:,.2f}"
            , 'ProbITM': "{:.0%}"
            , "num_puts": "{:,.0f}"
            , "current": "${:,.0f}"
            , 'change_from_strike': "{:.0%}"
        }).hide_index())
    print(f"""
funds= ${savings:,.0f} + ${brokerage_cash:,.0f} + ${brokerage_asset:,.0f} + ${adjustments:,.0f} =   ${funds:,.0f}
max_exposure= ${max_exposure:,.0f}
expected_exposure= ${expected_exposure:,.0f}
risk_aware_exposure= ${risk_aware_exposure:,.0f}
capital= ${capital:,.0f}
"""  )
    return capital

def gamble_suggest(target_date
                   , gamble_file
                   , exclude_current=True
                   , exclude_upcoming_earnings= True
                   , exclude = [] 
                   , min_stocks= 6
                   , **kwargs):
    if exclude_current:
        exclude += pd.read_csv(kwargs['put_obligation_file'])['stock'].to_list()
    if exclude_upcoming_earnings:
        upcoming_earnings = list(filter( lambda _ : not is_safe_to_write_put(_,target_date) , pd.read_csv(gamble_file)['stock'].to_list()))
        print(f"Earnings upcoming from {' , '.join(map(lambda _ : _ + ' on ' + earnings_date(_),  upcoming_earnings))} and therefore excluding.." )
        exclude += upcoming_earnings
        
    print(f"#trading days = {trading_days(target_date)}")
    capital= get_capital(target_date= target_date, **kwargs)
    df = pd.read_csv(gamble_file).drop_duplicates()
    df = df[~df['stock'].isin(exclude)]
    df = df[vectorize(option_exists)( df['stock'], target_date )]
    #df['current'] = df.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    df['current'] = vectorize(lambda _: stock_prices(_)['current'])(df['stock'])
    df['ideal_strike'] = df.apply(
        lambda r: strike(r['stock']
                    , trading_days(target_date)
                    , r['quality']
                    , r['distance_52weekhigh'] ), axis=1)
    
    df['strike'] = df.apply(
        lambda r: nearest_to(
            list([ _ for _ in chains(r['stock'])['PUT'] if _["expiry"] == target_date ][0]["strikes"].keys())  
            , r['ideal_strike']), axis=1)
    df['price'] = df.apply(lambda r: option_price('PUT', r['stock'], target_date, r['strike']), axis=1)
    df['price_efficiency'] = df.apply(lambda r: (r['price'] - max(r['strike'] - r['ideal_strike'],0)  ) / r['strike'] if r['strike'] > 0 else -1,axis=1)
    df = df.sort_values(by='price_efficiency', ascending=False).head(min_stocks)

    df['fund_alloc_prop'] = df['price_efficiency'] / df['price_efficiency'].sum()
    df['fund_alloc'] = df['fund_alloc_prop'] * capital

    df['num_puts'] = (df['fund_alloc'] / (df['strike'] * 100)).round()
    df['revenue'] = df['price'] * df['num_puts']*100
    print(f"\n\nTotal revenue = ${df['revenue'].sum():,.0f}")
    roi_from_weeklys = lambda zs: (lambda x: (1 + x[1] / x[0]) ** 52 - 1)(
        reduce(lambda acc, x: (acc[0] + x[0] * x[2], acc[1] + x[1] * x[2]), zs, (0, 0)))
    print(f"ROI = {roi_from_weeklys(df[['strike', 'price', 'num_puts']].to_records(index=False)):.0%}")
    print(f"Capital utilization = {(df['strike'] * df['num_puts']).sum() * 100 / capital:.0%}")

    return df["current,stock,strike,price,num_puts,revenue,price_efficiency,quality,fund_alloc_prop,fund_alloc,ideal_strike".split(',')].sort_values(
        by='price_efficiency', ascending=False).style.format(
        {
            "current": "${:,.0f}"
            , "ideal_strike": "${:,.1f}"
            , "strike": "${:,.1f}"
            , "price": "${:,.2f}"
            , "price_efficiency": "{:.1%}"
            , 'quality': "{:.0f}"
            , 'fund_alloc_prop': "{:.0%}"
            , 'fund_alloc': "${:,.0f}"
            , "num_puts": "{:,.0f}"
            , "revenue": "${:,.0f}"
        }).hide_index()

def exit_aggressive(portfolio_file, exclude_upcoming_earnings= True):
    df = pd.read_csv(portfolio_file)
    df['num_calls'] = df['quantity']//100
    df['out'] = vectorize( stock_prices )( df['stock'] )
    df['last_max'] = vectorize( lambda _ : _['last_max'] )( df['out'] )
    df['r'] = vectorize( lambda _ : _['rSigma']['high'][0] )( df['out'] )
    df['sigma'] = vectorize( lambda _ : _['rSigma']['high'][1] )( df['out'] )
    df['earnings_date'] = vectorize( earnings_date )( df['stock'] )
    
    df_curr = df[ ~df['call_strike'].isnull() ]
    df = df[ df['call_strike'].isnull() ]
    if not df_curr.empty:
        df_curr['days_to_expiry'] = vectorize( trading_days )( df_curr['call_expiry']  )
    
        df_curr['ProbITM'] = vectorize(tail_probability('higher'))( df_curr['stock']
                                                     , df_curr['call_strike']
                                                     , df_curr['days_to_expiry'] )
        display(df_curr["stock,quality,last_max,call_expiry,call_strike,call_price,num_calls,ProbITM,earnings_date".split(',')].sort_values(
            by='ProbITM', ascending=False).style.format(
            {
                "num_calls": "{:,.0f}"
                , "last_max": "${:,.2f}"
                , "call_strike": "${:,.1f}"
                , "call_price": "${:,.2f}"
                , 'ProbITM': "{:.0%}"
                , 'quality': "{:.0f}"
            }).hide_index())
    
    df['expiry'] = vectorize( lambda _ :   chains(_)["CALL"][0]["expiry"])(df['stock'])
    if exclude_upcoming_earnings:
        print(f"\nExcluding {','.join(df[~vectorize(is_safe_to_write_put)(df['stock'], df['expiry'] )]['stock'].values)} due to upcoming earnings")
        df = df[vectorize(is_safe_to_write_put)(df['stock'], df['expiry'] )]
    df['days_to_expiry'] = vectorize( trading_days )( df['expiry']  )
    
    df['ideal_strike'] = df['last_max'] *  exp((df['r'] + ( (4-df['quality']) / 2) * df['sigma']) * df['days_to_expiry'])
    df['strike'] = df.apply(
            lambda r: nearest_to(
                list([ _ for _ in chains(r['stock'])['CALL'] if _["expiry"] == r['expiry'] ][0]["strikes"].keys())  
                , r['ideal_strike']), axis=1)
    df['option_price'] = vectorize(option_price)( 'CALL', df['stock'], df['expiry'], df['strike'] )

    
    df['price_efficiency'] = df["option_price"]/( df['cost'] - df['strike'] )
    
    return df["stock,quality,last_max,cost,expiry,strike,option_price,num_calls,price_efficiency,earnings_date".split(',')].sort_values(
            by='price_efficiency', ascending=False).style.format(
            {
                "num_calls": "{:,.0f}"
                , 'quality': "{:.0f}"
                , "last_max": "${:,.2f}"
                , "strike": "${:,.1f}"
                , "option_price": "${:,.2f}"
                , "cost": "${:,.2f}"
                , "price_efficiency": "{:.1%}"
            }).hide_index()