import json
import time
import sys
from datetime import datetime
import ccxt

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

period_table = {'60':'1m',
                '180':'3m',
                '300':'5m',
                '900':'15m',
                '1800':'30m',
                '3600':'1h',
                '7200':'2h',
                '14400':'4h',
                '21600':'6h',
                '43200':'12h',
                '86400':'1d',
                '604800':'1w'}

# Possible Commands
PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume', 'returnOrderBook', 'returnTradeHistory', 'returnChartData',
                   'returnCurrencies', 'returnLoanOrders']

class Poloniex:
    def __init__(self, market, exchange, APIKey='', Secret=''):
        self.list_exchanges = {"kucoin":ccxt.kucoin(), "binance":ccxt.binance()}
        if market in self.list_exchanges.keys():
            self.market = self.list_exchanges[market]
            print("Training on", market)
        else:
            print("Market ",market," is not supported. Supported markets: ", self.list_exchanges.keys())
            time.sleep(5)
        if exchange in self.list_exchanges.keys():
            self.exchange = self.list_exchanges[exchange]
            print("Trade on", exchange)
        else:
            print("Exchange ",exchange," is not supported. Supported exchanges: ", self.list_exchanges.keys())
            time.sleep(5)
        #self.common = self.give_common_pairs(self.exchange,ccxt.poloniex())
        self.common = self.give_common_pairs(self.exchange, self.market)
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S": datetime.fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S": int(time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2: str(round(float(floatN) * 100, decimalP))+"%"
        self.eX = self.market
        self.eX.load_markets()

        # PUBLIC COMMANDS
        self.marketTicker = lambda x=0: self.api('returnTicker')
        self.marketVolume = lambda x=0: self.api('return24hVolume')
        self.marketStatus = lambda x=0: self.api('returnCurrencies')
        self.marketLoans = lambda coin: self.api('returnLoanOrders',{'currency':coin})
        self.marketOrders = lambda pair='all', depth=10:\
            self.api('returnOrderBook', {'currencyPair':pair, 'depth':depth})
        self.marketChart = lambda pair, period=day, start=time.time()-(week*1), end=time.time(): self.api('returnChartData', {'currencyPair':pair, 'period':period, 'start':start, 'end':end})
        self.marketTradeHist = lambda pair: self.api('returnTradeHistory',{'currencyPair':pair}) # NEEDS TO BE FIXED ON Poloniex

    #####################
    # Main Api Function #
    #####################
    def api_poloniex(self, command, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            ret = urlopen(Request(url + urlencode(args)))
            res = json.loads(ret.read().decode(encoding='UTF-8'))
            if command == 'returnChartData':
                res = [ticker for ticker in res if ticker['volume']>0]
            return res
        else:
            return False

    def api(self, command, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """

        if command == 'returnTicker':
            tickers_ccxt = self.eX.fetch_tickers()
            # print(tickers_ccxt, tickers_ccxt.keys())
            tickers = {}
            for symbol, data in tickers_ccxt.items():
                if "/" in symbol:
                    quote, base = str(symbol).split("/")
                    new_symbol = base + "_" + quote
                    tickers[new_symbol] = data
            # print(tickers, tickers.keys())
            return tickers
        elif command == "return24hVolume":
            tickers_ccxt = self.eX.fetch_tickers()
            # print(tickers_ccxt)
            tickers = {}
            for symbol, data in tickers_ccxt.items():
                if "/" in symbol:
                    quote, base = str(symbol).split("/")
                    new_symbol = base + "_" + quote
                    tickers[new_symbol] = data
            vol = {}
            for symbol, data in tickers.items():
                base, quote = str(symbol).split("_")
                vol[symbol] = {base: data["quoteVolume"], quote: data["baseVolume"]}
            #print(vol)
            return vol
        elif command == "returnChartData":
           # args = {'currencyPair': pair, 'period': period, 'start': start, 'end': end})
           chart = []
           base, quote = str(args["currencyPair"]).split("_")
           pair = quote+"/"+base
           period = int(args['period'])
           start = int(args["start"]*1000)
           end = int(args["end"]*1000)
           #start = start-start%(period*1000)
           #end = end-end%(period*1000)
           #limit = int(int(end-start)/(1000*period))
           #print(limit)
           #print("Fetching pair "+str(pair))
           #ohlcv = self.eX.fetch_ohlcv(symbol=pair, timeframe=period_table[str(period)], since=start,limit=1)
           # if ohlcv[0][0]>= end :
           #     return [{'date':0,'open':0,'high':0,'low':0,'close':0,
           #                 'volume':0,'quoteVolume':0,'weightedAverage':0}]
           # else:
           return self.get_full_chart(pair, period, start, end)

        else:
            return False

    def get_full_chart(self,pair, period, start, end): # checked only for binance
        # start and end needs to be in milliseconds, period in seconds, pair like "ETH/BTC"
        _period = 1000 * period
        _start = start - start % _period
        now = int(time.time()) * 1000
        latest_ticker = now - now % _period
        _end = min ( (end - end % _period) , latest_ticker ) # make sure end ticker exists and is full
        limit = max(1,int(int(_end - _start) / _period))
        chart = []
        keep_fetching = True
        while keep_fetching:
            ohlcv = self.eX.fetch_ohlcv(symbol=pair, timeframe=period_table[str(period)], since=_start, limit=limit)
            if not ohlcv: # case where exchange returns [] (poloniex)
                chart.append({'date': 0, 'open': 0, 'high': 0, 'low': 0, 'close': 0,
                          'volume': 0, 'quoteVolume': 0, 'weightedAverage': 0})
                keep_fetching = False
            else:
                if ohlcv[0][0] >= _end: # if exchange fetches only from first available date
                    chart.append({'date': 0, 'open': 0, 'high': 0, 'low': 0, 'close': 0,
                                  'volume': 0, 'quoteVolume': 0, 'weightedAverage': 0})
                    keep_fetching = False
                else:
                    for t in ohlcv:
                        chart.append({'date': int(t[0] / 1000), 'open': t[1], 'high': t[2], 'low': t[3], 'close': t[4],
                                      'volume': t[5]*t[4], 'quoteVolume': t[5], 'weightedAverage': t[4]})
                    last_date = ohlcv[-1][0]
                    if last_date == _end - _period:
                        keep_fetching = False
                    elif  last_date > _end - _period:
                        chart = [a for a in chart if a['date']<=last_date]
                        keep_fetching = False
                    else:
                        _start = last_date + _period
                        limit = max(1, int(int(_end - _start) / _period)) # in the case rare case where _start = end
        return chart


    def get_pairs(self,e):
        e.load_markets()
        pairs=list(e.markets.keys())
        return pairs

    def give_common_pairs(self,e1,e2):
        pairs_e1 = self.get_pairs(e1)
        pairs_e2 = self.get_pairs(e2)
        common = []
        for p1 in pairs_e1:
            for p2 in pairs_e2:
                if p1==p2:
                    common.append(p1)
        common = self.change_syntax(common)
        return common

    def change_syntax(self,pairs):
        new_pairs = []
        for symbol in pairs:
            quote, base = str(symbol).split("/")
            new_symbol = base + "_" + quote
            new_pairs.append(new_symbol)
        return new_pairs