#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:26:00 2021

@author: chrisvarghese
"""
from functools import partial
class OCOorder:
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


def new_cost(currentCost, sellPrice, numSold, numCurrent):
    numHeld = numCurrent - numSold
    return currentCost - (sellPrice - currentCost) * numSold / numHeld



def backr(spread, target=None, buy=1, edge=None, disc=0.5, prem=0.05, fee=0.02):
    assert target or edge, "target or edge needs to be specified"
    assert buy in [-1, 1], "buy has to be +1 or -1"
    t = target if target else edge * (1 + disc) ** (-buy)  # np.exp( - buy * np.log(1+disc) )
    z = nearest(spread)(t * (1 + buy * prem))
    return sorted([z + buy * 2 * spread, z + buy * spread, buy * (z - t), fee + buy * (z - t)], reverse=True)


def single(spread, target=None, buy=1, edge=None, disc=0.5, prem=0.05, fee=0.01):
    assert target or edge, "target or edge needs to be specified"
    assert buy in [-1, 1], "buy has to be +1 or -1"
    t = target if target else edge * (1 + disc) ** (-buy)
    z = nearest(spread)(t * (1 + buy * prem))
    return sorted([z, buy * (z - t), fee + buy * (z - t)], reverse=True)


def single_suggest(stock, buy, target, **kwargs):
    return {k: next(
        filter(
            lambda _: set(_[:1]).issubset(set(map(float, v.split(','))))
            , map(partial(single, buy=buy, target=target, **kwargs), (0.5 * _ for _ in range(1, 21)))), None)
        for k, v in chains(stock, 'PUT' if buy == 1 else 'CALL').items()}


def backr_suggest(stock, buy, target, **kwargs):
    return {k: next(
        filter(
            lambda _: set(_[:2]).issubset(set(map(float, v.split(','))))
            , map(partial(backr, buy=buy, target=target, **kwargs), (0.5 * _ for _ in range(1, 21)))), None)
        for k, v in chains(stock, 'PUT' if buy == 1 else 'CALL').items()}


def account_info(access_token):
    """ get access token from https://developer.tdameritrade.com/authentication/apis/post/token-0"""
    header = {"Authorization": 'Bearer ' + access_token}
    positions = requests.get(
        url=f"https://api.tdameritrade.com/v1/accounts/{config.account_id}"
        , headers=header
        , params={'fields': 'positions'}).json()['securitiesAccount']['positions']

    orders = requests.get(
        url=f"https://api.tdameritrade.com/v1/accounts/{config.account_id}/orders"
        , headers=header).json()

    transactions = requests.get(
        url=f"https://api.tdameritrade.com/v1/accounts/{config.account_id}/transactions"
        , headers=header).json()
    return positions, orders, transactions

def value_invest_suggest(wish_list_file):
    to_buy = pd.read_csv(wish_list_file)
    to_buy['current'] = to_buy.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    to_buy['Xprob'] = to_buy.apply(lambda row: StockPrices.FallBelowProb(row['stock'], row['price']), axis=1)
    to_buy['nto'] = to_buy.apply(lambda r: near_term_option(r['stock'], 1, r['price']), axis=1)
    return to_buy.sort_values('Xprob', ascending=False)

def near_term_option(stock, buy, target, fee=0.01):
    d = chains(stock, 'PUT' if buy == 1 else 'CALL')
    earliest_date = min((_ for _ in d.keys() if dt.datetime.fromisoformat(_) > dt.datetime.today()),
                        key=dt.datetime.fromisoformat)
    return earliest_date, sorted(
        [(k, (k - target) * buy + fee) for k in map(float, d[earliest_date].split(',')) ],
        key=lambda _: _[1])

def near_term_options(stock, buy=1):
    d = chains(stock, 'PUT' if buy == 1 else 'CALL')
    earliest_date = min((_ for _ in d.keys() if dt.datetime.fromisoformat(_) > dt.datetime.today()),
                        key=dt.datetime.fromisoformat)
    return earliest_date, sorted(map(float, d[earliest_date].split(',')))

"""
upper_bounds = lambda s, tau: StockPrices.get(s)['last_max'] * np.exp(
    (StockPrices.get(s)['rSigma']['high'][0] + 1 * StockPrices.get(s)['rSigma']['high'][1]) * tau)

def exit_orders(portfolio):
    to_sell = pd.DataFrame([(k.stock, k.target, k.quantity//100 ) for k in portfolio.Positions], columns=['stock', 'target','num_calls'])
    to_sell['current'] = to_sell.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    to_sell['Xprob'] = to_sell.apply(lambda row: StockPrices.RiseAboveProb(row['stock'], row['target']), axis=1)
    to_sell['nto'] = to_sell.apply(lambda r: near_term_option(r['stock'], -1, r['target']), axis=1)
    return to_sell.sort_values('Xprob', ascending=False).style.format(
        {
            "target": "${:,.2f}"
            , "current": "${:,.2f}"
            , "num_calls": "{:,.0f}"
            # , "price": "${:,.2f}"
            , "Xprob": "{:.0%}"
        }).hide_index()

def gamble_suggest_exit(target_date, portfolio):
    gamble = pd.DataFrame([(k.stock, k.target) for k in portfolio.Positions], columns=['stock', 'target'])
    gamble['current'] = gamble.apply(lambda row: StockPrices.get(row['stock'])['current'], axis=1)
    gamble['upper_bounds'] = gamble.apply(
        lambda r: nearest_to(map(float, chains(r['stock'], 'CALL').get(target_date, "0").split(',')),
                                   upper_bounds(r['stock'], trading_days(target_date))), axis=1)
    gamble['upper_bounds_price'] = gamble.apply(
        lambda r: call_price('mark')(r['stock'], target_date, r['upper_bounds']), axis=1)

    gamble['price_efficiency'] = gamble.apply(lambda r: r['upper_bounds_price'] / (r['target'] - r['upper_bounds']),
                                              axis=1)
    return gamble.sort_values(by='price_efficiency', ascending=False).style.format(
        {
            "current": "${:,.0f}"
            , "upper_bounds": "${:,.1f}"
            , "upper_bounds_price": "${:,.2f}"
            , "price_efficiency": "{:.1%}"

        }).hide_index()
        """

"""
@lru_cache(maxsize=None)
def chains(stock, option_type):
    extract = lambda d: {k.split(':')[0]: ','.join([i[-2:] == '.0' and i[:-2] or i for i in v.keys()]) for k, v in
                         d.items()}
    return extract(requests.get(url=f'https://api.tdameritrade.com/v1/marketdata/chains',
                         params={'apikey': config.client_id
                             , 'symbol': stock
                             , 'contractType': option_type
                                 }).json()[f'{option_type.lower()}ExpDateMap'])
"""