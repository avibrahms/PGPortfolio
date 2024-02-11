from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pgportfolio.marketdata.poloniex import Poloniex
from pgportfolio.tools.data import get_chart_until_success
import pandas as pd
from datetime import datetime
import logging
from pgportfolio.constants import *
import time
from threading import Thread
import numpy as np

class CoinList(object):
    def __init__(self, market,exchange,end, volume_average_days=1, volume_forward=0):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.CRITICAL)
        self._polo = Poloniex(market,exchange)
        # connect the internet to accees volumes
        self._volume_average_days = volume_average_days
        vol = self._polo.marketVolume()
        ticker = self._polo.marketTicker()
        len_vol = len(vol.items())
        pairs = np.array(["0123456789___"] * len_vol)
        coins = np.array(["0123456789___"] * len_vol)
        volumes = np.array([-1.0] * len_vol)
        prices = np.array([-1.0] * len_vol)
        ####################################################
        def get_volume(i,k,v):
            if k.startswith("BTC_") or k.endswith("_BTC"):
                pairs[i]=k
                for c, val in v.items():
                    if c != 'BTC':
                        if k.endswith('_BTC'):
                            coins[i] = 'reversed_' + c
                            prices[i] = 1.0 / float(ticker[k]['last']) if ticker[k]['last'] else 0
                        else:
                            coins[i] = c
                            prices[i] = float(ticker[k]['last']) if ticker[k]['last'] else 0
                    else:
                        volumes[i] = self.__get_total_volume(pair=k, global_end=end,
                                                               days=volume_average_days,
                                                               forward=volume_forward)
        ######################################################
        logging.info("Selecting top volumes coins online from %s to %s" % (datetime.fromtimestamp(end-(DAY*volume_average_days)-
                                                                                  volume_forward).
                                                           strftime('%Y-%m-%d %H:%M'),
                                                           datetime.fromtimestamp(end-volume_forward).
                                                           strftime('%Y-%m-%d %H:%M')))
        threads = []
        for i, (k, v) in enumerate(vol.items()):
            t = Thread(target=get_volume, args=(i,k,v))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        coins = coins[coins!="0123456789___"]
        pairs = pairs[pairs != "0123456789___"]
        volumes = volumes[volumes != -1]
        prices = prices[prices != -1]
        logging.debug(coins,pairs,volumes,prices)

        self._df = pd.DataFrame({'coin': coins, 'pair': pairs, 'volume': volumes, 'price':prices})
        self._df = self._df.set_index('coin')

        self.logger.setLevel(logging.DEBUG)

    @property
    def allActiveCoins(self):
        return self._df

    #@property
    #def allCoins(self):
     #   return self._polo.marketStatus().keys()

    @property
    def polo(self):
        return self._polo

    def get_chart_until_success(self, pair, start, period, end):
        return get_chart_until_success(self._polo, pair, start, period, end)

    # get several days volume
    def __get_total_volume(self, pair, global_end, days, forward):
        start = global_end-(DAY*days)-forward
        end = global_end-forward
        chart = self.get_chart_until_success(pair=pair, period=DAY, start=start, end=end)
        result = 0
        for one_day in chart:
            if pair.startswith("BTC_"):
                result += one_day['volume']
            else:
                result += one_day["quoteVolume"]
        return result


    def topNVolume(self, n=5, order=True, minVolume=0):
        #minVolume = self._volume_average_days * 600
        if minVolume == 0:
            r = self._df#.loc[self._df['price'] > 2e-9]
            p = r['pair']
            r = r[p.isin(self._polo.common)]
            r = r[~p.isin(['BTC_BNB'])]
            r = r[r['volume'] >= 0]
            if r.shape[0] < n:
                print("Number of coins needs to be inferior or equal to ",r.shape[0],". Please change it.")
                time.sleep(5)
                return
            res = r.sort_values(by='volume', ascending=False)[:n]
            #print(r)
            if order:
                print(res)
                return res
            else:
                res = res.sort_index()
                print(res)
                return res
        else:
            res = self._df[self._df.volume >= minVolume]
            print(res)
            return res

