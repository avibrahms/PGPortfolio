import ccxt

e = ccxt.poloniex()
e.load_markets()
pairs=list(e.markets.keys())
print(len(pairs,pairs)
MILLI = 1000
MIN = 60 * MILLI
HOUR = MIN * 60
DAY = HOUR * 24
WEEK = DAY * 7
MONTH = WEEK * 30
YEAR = MONTH * 12


#symbols = binance.symbols
#print(markets)
#print(symbols)
cur = list(e.currencies.keys())
print(len(cur),"currencies:\n",cur)

ohlcv = e.fetch_ohlcv('BTC/USDT','1d',since=1230049600000,limit=3000)
print(int(ohlcv[-1][0]-ohlcv[0][0])/HOUR)
print(len(ohlcv))
print(ohlcv)

tickers = e.fetchTickers()
print(tickers)

#ticker = e.fetchTicker('BTC/USDT')
#print(ticker)
print(list(tickers.keys()))

#print(huobi.id, huobi.load_markets())

#print("R2 "+ eX.fetch_order_book('ETH/BTC'))
#print("R3 "+ eX.fetch_ticker('ETH/BTC'))
#print(huobi.fetch_trades('LTC/CNY'))

#print(exmo.fetch_balance())

# sell one ฿ for market price and receive $ right now
#print(exmo.id, exmo.create_market_sell_order('BTC/USD', 1))

# limit buy BTC/EUR, you pay €2500 and receive ฿1  when the order is closed
#print(exmo.id, exmo.create_limit_buy_order('BTC/EUR', 1, 2500.00))

# pass/redefine custom exchange-specific order params: type, amount, price, flags, etc...
#kraken.create_market_buy_order('BTC/USD', 1, {'trading_agreement': 'agree'})