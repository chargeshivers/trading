import pandas as pd
import numpy as np
import scipy.stats as sp
import datetime as dt
import heapq
from config import client_id
import requests
from collections import defaultdict
from IPython.display import display
from functools import partial

quote = lambda s: requests.get(url=f'https://api.tdameritrade.com/v1/marketdata/{s}/quotes?',
                               params={'apikey': client_id}).json().get(s, defaultdict(int))

extract = lambda d: {k.split(':')[0]: ','.join([i[-2:] == '.0' and i[:-2] or i for i in v.keys()]) for k, v in
                     d.items()}
chains = lambda s, t: extract(requests.get(url=f'https://api.tdameritrade.com/v1/marketdata/chains',
                                           params={'apikey': client_id
                                               , 'symbol': s
                                               , 'contractType': t
                                                   }).json()[f'{t.lower()}ExpDateMap'])

history = lambda s: requests.get(
    f"https://api.tdameritrade.com/v1/marketdata/{s}/pricehistory"
    , params={
        'apikey': client_id
        , 'periodType': 'year'
        , 'period': 1
        , 'frequencyType': 'daily'
        , 'frequency': 1
    }).json()['candles']
extremes = lambda e: lambda s: list(map(lambda _: _[e], history(s)))
deltas = lambda xs: (a - b for a, b in zip(xs, xs[1:]))

floor = lambda b: lambda x: (x // b) * b
nearest = lambda b: lambda x: floor(b)(x) + (b if x - floor(b)(x) >= b / 2 else 0)


def backr(spread, target=None, buy=1, edge=None, disc=0.5, prem=0.05, fee=0.02):
    assert target or edge, "target or edge needs to be specified"
    assert buy in [-1, 1], "buy has to be +1 or -1"
    t = target if target else edge * (1 + disc) ** (-buy)  # np.exp( - buy * np.log(1+disc) )
    z = nearest(spread)(t * (1 + buy * prem))
    return sorted([z + buy * 2 * spread, z + buy * spread, buy * (z - t), fee + buy * (z - t)], reverse=True)


def backr_suggest(stock, buy, target):
    return {k: next(
        filter(
            lambda _: set(_[:2]).issubset(set(map(float, v.split(','))))
            , map(partial(backr, buy=buy, target=target), (0.5 * _ for _ in range(1, 21)))))
        for k, v in chains(stock, 'PUT' if buy == 1 else 'CALL').items()}


class StockPrices:
    data = {}
    tradingDay = int(dt.date.today().strftime('%Y%m%d'))

    @classmethod
    def get(self, stock, lookback_window=250):
        #nullReplace = lambda x: x if x else ss["latestPrice"]
        if stock not in self.data:
            ss = quote(stock)

            self.data[stock] = {}
            self.data[stock]["current"] = ss['lastPrice']  # ss['latestPrice']
            #nullReplace = lambda x: x if x else ss["lastPrice"]
            self.data[stock]["last_min"] = ss['lowPrice']  # nullReplace(ss["low"])
            self.data[stock]["last_max"] = ss['highPrice']  # nullReplace(ss["high"])
            self.data[stock]["52weekHigh"] = ss['52WkHigh']  # ss['week52High']
            self.data[stock]["52weekLow"] = ss['52WkLow']  # ss['week52Low']

            self.data[stock]["rSigma"] = {}

            for e in ["low", "high"]:
                self.data[stock]["rSigma"][e] = sp.cauchy.fit(list(deltas(list(map(np.log, extremes(e)(stock)[::-1])))))

        return self.data[stock]

    @classmethod
    def PredictedMin(self, stock):
        return self.get(stock)["last_min"] * np.exp(self.get(stock)["rSigma"]["low"][0])

    @classmethod
    def PredictedMax(self, stock):
        return self.get(stock)["last_max"] * np.exp(self.get(stock)["rSigma"]["high"][0])

    @classmethod
    def RiseAboveProb(self, stock, price):
        r, sigma = self.get(stock)["rSigma"]["high"]
        return 1 - sp.cauchy.cdf((-r + np.log(price / self.get(stock)["last_max"])) / sigma)

    @classmethod
    def FallBelowProb(self, stock, price):
        r, sigma = self.get(stock)["rSigma"]["low"]
        # return sp.cauchy.cdf( (  -r + np.log( price / self.get(stock)["last_min"] )  ) / sigma  )
        try:
            return sp.cauchy.cdf((-r + np.log(price / self.get(stock)["last_min"])) / sigma)
        except ZeroDivisionError:
            return 1


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
        # return round(  (self.price - StockPrices.get(self.stock)["52weekLow"])/(StockPrices.get(self.stock)["52weekHigh"] - StockPrices.get(self.stock)["52weekLow"]),2)
        try:
            return round((self.price - StockPrices.get(self.stock)["52weekLow"]) / (
                        StockPrices.get(self.stock)["52weekHigh"] - StockPrices.get(self.stock)["52weekLow"]), 2)
        except ZeroDivisionError:
            return 1

    def ExecutionProbability(self, otype="buy"):
        if otype == "buy":
            return StockPrices.FallBelowProb(self.stock, self.price)
        else:
            return StockPrices.RiseAboveProb(self.stock, self.price)


class OCOorder(object):
    def __init__(self, orders):
        self.orders = orders

    def __str__(self):
        return " | ".join([str(k) for k in self.orders])

    def cost(self):
        return max([k.cost() for k in self.orders])

    def wastage(self):
        return round(1 - min([k.cost() for k in self.orders]) / max([k.cost() for k in self.orders]), 2)

    def DistanceToTarget(self):
        return min([k.DistanceToTarget() for k in self.orders])

    def Tseparation(self):
        return abs(self.orders[0].DistanceToTarget() - self.orders[1].DistanceToTarget())

    def mergeOrders(self):
        biggerOrder = max(self.orders, key=lambda x: x.cost())
        smallerOrder = min(self.orders, key=lambda x: x.cost())
        sharesFromBiggerOrderToCombine = round(smallerOrder.cost() / biggerOrder.price)
        return (OCOorder(
            [
                smallerOrder
                , Order(biggerOrder.stock, biggerOrder.price, sharesFromBiggerOrderToCombine)
            ]
        )
        , Order(biggerOrder.stock, biggerOrder.price, biggerOrder.quantity - sharesFromBiggerOrderToCombine)
        )


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
            # self.Positions = [ Position(r["stock"], r["cost"], r["quantity"]) for i,r in pd.read_csv(File).iterrows()]
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

    def DisplayStats(self, sortby="gain", ascending=False, columns=None):
        df = pd.DataFrame([{"stock": p.stock
                               , "cost": p.price
                               , "quantity": p.quantity
                               , "investment": p.price * p.quantity
                               , "current": StockPrices.get(p.stock)['current']
                               , "gain": p.Gain()
                               , "Pos52": Order(p.stock, p.price).PositionIn52weekWindow()
                               , "target": p.target
                            # , "CallStrike" : np.floor(p.price*1.5*0.95)
                            # , "CallPrice" : p.price*1.5 - np.floor(p.price*1.5*0.95)
                               , "CallStrike": np.floor(p.target * 0.95)
                               , "CallPrice": p.target - np.floor(p.target * 0.95)
                            } for p in self.Positions])
        df["concentration"] = df["investment"] / df["investment"].sum()
        print("Total investment = " + "${:,.0f}".format(self.Cost()))
        print("Total Gain = " + "{:.1%}".format(self.Gain()))
        if not columns:
            columns = ["stock", "cost", "Pos52", "quantity", "investment", "concentration", "current", "gain", "target",
                       "CallStrike", "CallPrice"]
        # return df[columns].sort_values(sortby, ascending = ascending)
        display(df[columns].sort_values(sortby, ascending=ascending).style.format({
            "current": "${:,.2f}"
            , "cost": "${:,.2f}"
            , "investment": "${:,.0f}"
            , "gain": "{:.0%}"
            , "concentration": "{:.0%}"
            , "Pos52": "{:.0%}"
            , "target": "${:,.2f}"
            , "CallStrike": "${:,.2f}"
            , "CallPrice": "${:,.2f}"}).hide_index())


class MyHeap:
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self._data)[1]


class OrderBook:
    def __init__(self, IndividualOrders):
        self.IndividualOrders = MyHeap(initial=[Order(*k) for k in IndividualOrders], key=lambda x: -x.cost())
        self.OCOorders = []

    def cost(self):
        return sum([k.cost() for k in self.OCOorders] + [k[1].cost() for k in self.IndividualOrders._data])

    def __str__(self):
        outDf = pd.DataFrame()

        outDf = pd.DataFrame([{"order": str(k), "cost": k.cost(),
                               "wastage": k.wastage()}
                              for k in self.OCOorders])  # [["order","cost","DistanceToTarget","Tseparation","wastage"]]
        outDf = outDf.append([{"order": str(k[1]), "cost": k[1].cost(),
                               "wastage": 0}
                              for k in self.IndividualOrders._data])
        return outDf.to_string()

    def compress(self, AvailableFunds=None):
        print(f"cost = {self.cost()}")
        if AvailableFunds == None:
            AvailableFunds = self.cost() / 2 + 1
        if self.cost() >= AvailableFunds * 2:
            print(
                f"The total cost of orders is ${self.cost()}, which is more than twice the available funds of {AvailableFunds}")
            return

        while self.cost() > AvailableFunds and len(self.IndividualOrders._data) > 1:
            # print(self)
            largestOrder = self.IndividualOrders.pop()
            secondLargestOrder = self.IndividualOrders.pop()
            newOCOorder, newIndividualOrder = OCOorder([largestOrder, secondLargestOrder]).mergeOrders()
            self.OCOorders.append(newOCOorder)
            if newIndividualOrder.quantity > 0:
                self.IndividualOrders.push(newIndividualOrder)
            if max([k[1].cost() for k in self.IndividualOrders._data]) > AvailableFunds - sum(
                    [k.cost() for k in self.OCOorders]):
                AvailableFunds *= 1.01
                print(
                    f"Impossible to compress further with available funds; trying with AvailableFunds = {AvailableFunds}")

        print("compression complete")
        print(f"cost = {self.cost()}")
        print(self)
        return


class WishList:
    def __init__(self, Wishes=None, File=None, sell=False):
        """
        Wishes should be a list of (stock,price) pairs
        """
        if Wishes != None:
            self.df = pd.DataFrame([{"stock": s, "price": p} for s, p in Wishes])
        elif File != None:
            self.df = pd.read_csv(File)
        else:
            print("Error")
        self.sell = sell

    def DisplayStats(self, sortby="Xprob", ascending=False,
                     columns=["stock", "price", "suggPrice", "LastMin", "LastMax", "Xprob", "High52", "Pos52",
                              "suggXprob", "putStrike", "putPrice"]):
        self.df["LastMin"] = self.df.apply(lambda row: StockPrices.get(row["stock"])["last_min"], axis=1)
        self.df["LastMax"] = self.df.apply(lambda row: StockPrices.get(row["stock"])["last_max"], axis=1)
        if self.sell:
            self.df["Xprob"] = self.df.apply(
                lambda row: Order(row["stock"], row["price"]).ExecutionProbability(otype="sell"), axis=1)
        else:
            self.df["Xprob"] = self.df.apply(lambda row: Order(row["stock"], row["price"]).ExecutionProbability(),
                                             axis=1)
        self.df["High52"] = self.df.apply(lambda row: Order(row["stock"], row["price"]).DistanceFrom52weekHigh(),
                                          axis=1)
        self.df["Pos52"] = self.df.apply(lambda row: Order(row["stock"], row["price"]).PositionIn52weekWindow(), axis=1)
        self.df["suggPrice"] = self.df.apply(lambda row: StockPrices.get(row["stock"])["52weekHigh"] / 1.5, axis=1)
        self.df["suggXprob"] = self.df.apply(lambda row: Order(row["stock"], row["suggPrice"]).ExecutionProbability(),
                                             axis=1)
        self.df["putStrike"] = self.df.apply(lambda row: np.ceil(row["price"] * 1.05), axis=1)
        self.df["putPrice"] = self.df.apply(lambda row: np.ceil(row["price"] * 1.05) - row["price"], axis=1)

        # return self.df[columns].sort_values(sortby, ascending = ascending)

        display(self.df[columns]
                .sort_values(sortby, ascending=ascending)
                .style.format(
            {
                "price": "${:,.2f}"
                , "LastMin": "${:,.2f}"
                , "LastMax": "${:,.2f}"
                , "Xprob": "{:.0%}"
                , "High52": "{:.0%}"
                , "Pos52": "{:.0%}"
                , "suggPrice": "${:,.2f}"
                , "suggXprob": "{:.0%}"
                , "putStrike": "${:,.2f}"
                , "putPrice": "${:,.2f}"
            }).hide_index())


def newCost(currentCost, sellPrice, numSold, numCurrent):
    numHeld = numCurrent - numSold
    return currentCost - (sellPrice - currentCost) * numSold / numHeld


"""
            try:
                df = xf.get_historical_data(stock, end= dt.datetime.today(), start= dt.datetime.today() + dt.timedelta(-365), output_format = "pandas" )
                
                decay = 0.01**(1/lookback_window)
                
                df = df.reset_index()
                df['log_ret_min'] = np.log(df["low"]) - np.log(df["low"].shift(1))
                df['log_ret_max'] = np.log(df["high"]) - np.log(df["high"].shift(1))
                df = df[-lookback_window:]
                df['maxIndex']=df.index.max()
                df['weight'] = (decay**(-lookback_window)) * (decay**(df['maxIndex']-df.index))
                for x in ["low","high"]:
                    weighted_sample =[ k for k, i in df[ [f"log_ret_{x}", "weight"] ].values for _ in range(int(np.floor(i)))   ] 
                    self.data[stock]["rSigma"][x] = sp.cauchy.fit( weighted_sample )
            #except KeyError or ValueError or UnicodeDecodeError or IEXQueryError:
            except:
                print(f"unable to get historical data for {stock}")
"""
