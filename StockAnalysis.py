import pandas as pd
from numpy import sqrt, log, exp, floor, busday_count, vectorize
import scipy.stats as sp
import datetime as dt
import config
import requests
from IPython.display import display
from functools import reduce, lru_cache
from utility import logg, nearest_to, deltas, retry

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

class StockPrices:
    data = {}
    tradingDay = int(dt.date.today().strftime('%Y%m%d'))

    @classmethod
    def get(self, stock, lookback_window=250):
        extremes = lambda extremity: lambda stock: [_[extremity] for _ in history(stock)]
        if stock not in self.data:
            alias = {'lastPrice': 'current', 'lowPrice': 'last_min', 'highPrice': 'last_max', '52WkHigh': '52weekHigh',
                     '52WkLow': '52weekLow'}
            self.data[stock] = {alias[k]: v for k, v in quote(stock).items() if k in alias}
            for main, alt in {'last_min': 'low', 'last_max': 'high'}.items():
                if self.data[stock][main] == 0:
                    self.data[stock][main] = extremes(alt)(stock)[-1]
            self.data[stock]["rSigma"] = {e: sp.cauchy.fit(list(deltas(list(map(log, extremes(e)(stock)[::-1])))))
                                          for e in ["low", "high"]}
            
        return self.data[stock]

    @classmethod
    def PredictedMin(self, stock):
        return self.get(stock)["last_min"] * exp(self.get(stock)["rSigma"]["low"][0])

    @classmethod
    def PredictedMax(self, stock):
        return self.get(stock)["last_max"] * exp(self.get(stock)["rSigma"]["high"][0])

    @classmethod
    def RiseAboveProb(self, stock, price, days= 1):
        r, sigma = self.get(stock)["rSigma"]["high"]
        return 1 - sp.cauchy.cdf((-r*days + log(price / self.get(stock)["last_max"])) / (sigma*days))

    @classmethod
    def FallBelowProb(self, stock, price, days= 1):
        r, sigma = self.get(stock)["rSigma"]["low"]
        return sp.cauchy.cdf((-r*days + log(price / self.get(stock)["last_min"])) / (sigma*days))

class Order:
    def __init__(self, stock, price, quantity=0):
        self.stock = stock
        self.price = price
        self.quantity = quantity

    def __lt__(self, other):
        return self.price < other.price

    def __str__(self):
        return f"{self.stock}{self.price}x{self.quantity}"

    def cost(self):
        return self.price * self.quantity

    def DistanceToTarget(self):
        return round(StockPrices.get(self.stock)["last_min"] / self.price - 1, 2)

    def DistanceFrom52weekHigh(self):
        return round(StockPrices.get(self.stock)["52weekHigh"] / self.price - 1, 2)

    def PositionIn52weekWindow(self):
        return round((self.price - StockPrices.get(self.stock)["52weekLow"]) / (
                    StockPrices.get(self.stock)["52weekHigh"] - StockPrices.get(self.stock)["52weekLow"]), 2)

    def ExecutionProbability(self, otype="buy"):
        if otype == "buy":
            return StockPrices.FallBelowProb(self.stock, self.price)
        else:
            return StockPrices.RiseAboveProb(self.stock, self.price)

class Position(Order):
    def __init__(self, stock, price, quantity, target):
        self.target = (target if target > 0 else price * 1.5)
        Order.__init__(self, stock, price, quantity)

    def Gain(self):
        return StockPrices.get(self.stock)["current"] / self.price - 1

class Portfolio:
    def __init__(self, Positions=None, File=None):
        if Positions != None:
            self.Positions = Positions
        elif File != None:
            self.Positions = [Position(r["stock"], r["cost"], r["quantity"], r["target"]) for i, r in
                              pd.read_csv(File).iterrows()]
        else:
            print("Error")

    def Cost(self):
        return sum([k.price * k.quantity for k in self.Positions])

    def Gain(self):
        return sum([k.price * k.quantity * k.Gain() for k in self.Positions]) / sum(
            [k.price * k.quantity for k in self.Positions])

    def recordState(self, fileName):
        with open(fileName, "a") as myfile:
            myfile.write(f"{StockPrices.tradingDay},{self.Cost():.0f},{self.Gain():.4f}\n")

    #def to_df(self):
    def DisplayStats(self, sortby="gain", ascending=False, columns=None):
        df = pd.DataFrame([{"stock": p.stock
                               , "cost": p.price
                               , "quantity": p.quantity
                               , "investment": p.price * p.quantity
                               , "current": StockPrices.get(p.stock)['current']
                               , "gain": p.Gain()
                               , "Pos52": Order(p.stock, p.price).PositionIn52weekWindow()
                               , "target": p.target
                               #, "CallStrike": floor(p.target * 0.95)
                               #, "CallPrice": p.target - floor(p.target * 0.95)
                            } for p in self.Positions])
        df["concentration"] = df["investment"] / df["investment"].sum()
        print("\nTotal investment = " + "${:,.0f}".format(self.Cost()))
        print("\nTotal Gain = " + "{:.1%}".format(self.Gain()))
        if not columns:
            columns = ["stock", "cost", "Pos52", "quantity", "investment", "concentration", "current", "gain", "target"
                    #   , "CallStrike", "CallPrice"
                      ]
        return df[columns].sort_values(sortby, ascending=ascending).style.format({
            "current": "${:,.2f}"
            , "cost": "${:,.2f}"
            , "investment": "${:,.0f}"
            , "gain": "{:.0%}"
            , "concentration": "{:.0%}"
            , "Pos52": "{:.0%}"
            , "target": "${:,.2f}"
        #    , "CallStrike": "${:,.2f}"
         #   , "CallPrice": "${:,.2f}"
        }).hide_index()


@lru_cache(maxsize=None)
def strike(stock, days_to_expiry, quality):
    return StockPrices.get(stock)['last_min'] * exp(
    (StockPrices.get(stock)['rSigma']['low'][0] - (quality / 2) * StockPrices.get(stock)['rSigma']['low'][1]) * days_to_expiry)

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
    brokerage_asset = ( lambda d : (vectorize( StockPrices.RiseAboveProb  )( d['stock'], d['target'], trading_days(target_date) )*d["target"]*d["quantity"]).sum() )( pd.read_csv(portfolio_file)  )
    funds = savings + brokerage_cash + adjustments + brokerage_asset
    put_obligation = pd.read_csv(put_obligation_file)
    put_obligation['price'] = put_obligation.apply( 
        lambda row : sum(float(_) for _ in row['price'].split('+')) , axis=1)
    put_obligation['ProbITM'] = put_obligation.apply( 
        lambda row : StockPrices.FallBelowProb(row['stock'],row['strike'], trading_days(target_date) ) , axis=1)
    put_obligation['current'] = put_obligation.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    max_exposure = (put_obligation['strike']*put_obligation['num_puts']).sum()*100
    expected_exposure = (put_obligation['strike']*put_obligation['num_puts']*put_obligation['ProbITM']).sum()*100
    risk_aware_exposure= (1-risk_apetite)*max_exposure + risk_apetite*expected_exposure
    capital = funds - risk_aware_exposure
    print('\n')
    display(put_obligation.sort_values('ProbITM', ascending=False).style.format(
        {
            "strike": "${:,.1f}"
            , "price": "${:,.2f}"
            , 'ProbITM': "{:.0%}"
            , "num_puts": "{:,.0f}"
            , "current": "${:,.0f}"

        }).hide_index())
    print(f"""
funds= ${funds:,.0f}
max_exposure= ${max_exposure:,.0f}
expected_exposure= ${expected_exposure:,.0f}
risk_aware_exposure= ${risk_aware_exposure:,.0f}
capital= ${capital:,.0f}

"""  )
    return capital

def gamble_suggest(target_date
                   , gamble_file
                   , exclude_current=True
                   , exclude = [] 
                   , min_stocks= 4
                   , **kwargs):
    if exclude_current:
        exclude += pd.read_csv(kwargs['put_obligation_file'])['stock'].to_list()
        
    print(f"#trading days = {trading_days(target_date)}")
    capital= get_capital(target_date= target_date, **kwargs)
    gamble = pd.read_csv(gamble_file, usecols=['stock','quality']).drop_duplicates()
    gamble = gamble[~gamble['stock'].isin(exclude)]
    gamble = gamble[vectorize(option_exists)( gamble['stock'], target_date )]
    gamble['current'] = gamble.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    gamble['strike'] = gamble.apply(
        lambda r: nearest_to(
            list([ _ for _ in chains(r['stock'])['PUT'] if _["expiry"] == target_date ][0]["strikes"].keys())  
            ,strike(r['stock']
                         , trading_days(target_date), r['quality'])), axis=1)
    gamble['price'] = gamble.apply(lambda r: option_price('PUT', r['stock'], target_date, r['strike']), axis=1)
    
    gamble['price_efficiency'] = gamble.apply(lambda r: r['price'] / r['strike'] if r['strike'] > 0 else -1,
                                              axis=1)
    gamble = gamble.sort_values(by='price_efficiency', ascending=False).head(min_stocks)

    gamble['fund_alloc_prop'] = gamble['price_efficiency'] / gamble['price_efficiency'].sum()
    gamble['fund_alloc'] = gamble['fund_alloc_prop'] * capital

    gamble['num_puts'] = (gamble['fund_alloc'] / (gamble['strike'] * 100)).round()
    gamble['revenue'] = gamble['price'] * gamble['num_puts']*100
    print(f"\n\nTotal revenue = ${gamble['revenue'].sum():,.0f}")
    roi_from_weeklys = lambda zs: (lambda x: (1 + x[1] / x[0]) ** 52 - 1)(
        reduce(lambda acc, x: (acc[0] + x[0] * x[2], acc[1] + x[1] * x[2]), zs, (0, 0)))
    print(f"ROI = {roi_from_weeklys(gamble[['strike', 'price', 'num_puts']].to_records(index=False)):.0%}")
    print(f"Capital utilization = {(gamble['strike'] * gamble['num_puts']).sum() * 100 / capital:.0%}")

    return gamble["current,stock,strike,price,num_puts,revenue,price_efficiency,quality,fund_alloc_prop,fund_alloc".split(',')].sort_values(
        by='price_efficiency', ascending=False).style.format(
        {
            "current": "${:,.0f}"
            , "strike": "${:,.1f}"
            , "price": "${:,.2f}"
            , "price_efficiency": "{:.1%}"
            , 'quality': "{:.0f}"
            , 'fund_alloc_prop': "{:.0%}"
            , 'fund_alloc': "${:,.0f}"
            , "num_puts": "{:,.0f}"
            , "revenue": "${:,.0f}"

        }).hide_index()

def exit_orders(portfolio):
    out=[]
    for position in portfolio.Positions:
        expiry = chains(position.stock)["CALL"][0]["expiry"]
        strikes = chains(position.stock)["CALL"][0]['strikes']
        K, p, c = max(
                        [(strike, StockPrices.RiseAboveProb(position.stock, strike, trading_days(expiry)), call_price) for strike, call_price in strikes.items()]
                        , key = lambda _ : (_[0]*_[1]+_[2])*( _[0] +_[2] > position.target )
        )
        out.append( 
            { "num_calls": position.quantity//100
             , "current" : StockPrices.get(position.stock)['current']
             , "target" : position.target
             , "stock" : position.stock
             , "strike" : K
             , "expiry" : expiry
             , "call_price" : c
             , "Xprob" : p}  if (K + c > position.target and c > 0) else  
            {"num_calls": position.quantity//100
             , "current" : StockPrices.get(position.stock)['current']
             , "target" : position.target
             , "stock" : position.stock
             , "expiry" : expiry}  )
    return pd.DataFrame(out).sort_values(by='Xprob', ascending=False).style.format(
            {
                "num_calls": "{:,.0f}"
                ,"current" : "${:,.2f}"
                ,"target" : "${:,.2f}"
                , "strike": "${:,.1f}"
                , "call_price": "${:,.2f}"
                , "Xprob" : "{:.0%}"
            }).hide_index()
