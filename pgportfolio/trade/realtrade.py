from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.poloniex import Poloniex
from pgportfolio.tools.trade import calculate_pv_after_commission
from threading import Thread
import logging
import time
from datetime import datetime
import sys
import json
if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
else:
    from urllib2 import Request, urlopen

class RealTrade(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        # if agent_type == "nn":
        #     data_matrices = self._rolling_trainer.data_matrices
        # elif agent_type == "traditional":
        #     config["input"]["feature_number"] = 1
        #     data_matrices = DataMatrices.create_from_config(config)
        # else:
        #     raise ValueError()
        #self.__test_set = data_matrices.get_test_set()
        #self.__test_length = self.__test_set["X"].shape[0]
        self.first_time = True
        self.unused_btc = config["input"]["unused_btc"]
        self.unused_usdt = config["input"]["unused_usdt"]
        self._total_steps = config["training"]["steps"]
        self.feature_number = config["input"]["feature_number"]
        self._period = config["input"]["global_period"]
        self.trading_granularity = config["input"]["trading_granularity"]
        if self.trading_granularity < 60 :
            raise Exception("Trading granularity should be greater or equal to 60 seconds")
        elif self.trading_granularity > self._period :
            raise Exception("Trading granularity should be lower than trading period")
        elif self._period % self.trading_granularity != 0 :
            raise Exception("Trading period should be a multiple of granularity")
        # elif self.trading_granularity != 60 and self.trading_granularity != self._period:
        #     raise Exception("Trading granularity not yet supported, should be 60 seconds or equal to trading period")
        self.interval = config["input"]["global_period"]
        self._exchange = config["input"]["exchange"]
        self.polo = Poloniex(config["input"]["market"],config["input"]["exchange"])
        self.list_exchanges = self.polo.list_exchanges
        self.exchange = self.list_exchanges[self._exchange]
        self.simulation = config["input"]["simulation"]
        self.last_qtty_commission_coin = 0.0
        self.COMMISSION_COIN = 'BNB'
        if not self.simulation:
            self.connect_api()
            self.balance = self.exchange.fetch_balance()
            self.last_qtty_commission_coin = self.balance["free"][self.COMMISSION_COIN]
        self.config = config
        self.window = config["input"]["window_size"]
        self.min_usd_trade = config["input"]["min_usd_trade"]
        self.__test_pv = 1.0
        self.__test_pc_vector = []
        self.coin_number = config["input"]["coin_number"]
        self._last_omega = np.zeros((self._coin_number + 1,))  # / (self._coin_number+1)
        self._last_omega[0] = 1.0
        self._last_omega_before_commission = self._last_omega.copy()
        self.pv_after_next_commission = self._last_omega[0]
        self.current_bitcoin_price = -1 # to be determined while fetching each period
        self.prices = np.concatenate((np.ones(1),np.zeros(self.coin_number)))         # to be determined later
        self.qtties = np.concatenate((np.ones(1),np.zeros(self.coin_number)))        # to be determined later
        self.next_qtties = np.zeros(self.coin_number+1)        # to be determined later
        self.current_price = np.ones(self.coin_number+1)
        self._total_capital = config["input"]["initial_BTC"]
        self.last_total_capital = self._total_capital
        self._total_capital_before_slippage = self._total_capital
        self.slippage = 1.0
        self.total_average_slippage = 1.0
        self.total_accumulated_slippage = 1.0
        self.out_of_sync = False
        self.total_change = 1.0
        self.change = 1.0
        self.qtty_commission_coin = 0.0
        self.latest_commission_coin_price = 0.0
        self.fetching_time = 1.0
        self.start = 1.0
        # self.__rounding = 0.0
        # self._average_rounding = 0.0
        self.average_market_change = 1.0
        self.average_change = 1.0
        self.best_model_change = 1.0
        self.logger = logging.getLogger()
        self.period_table = {'60': '1m',
                             '180': '3m',
                             '300': '5m',
                             '900': '15m',
                             '1800': '30m',
                             '3600': '1h',
                             '7200': '2h',
                             '14400': '4h',
                             '21600': '6h',
                             '43200': '12h',
                             '86400': '1d',
                             '604800': '1w'}
        if str(self.trading_granularity) not in self.period_table:
            raise Exception("Trading granularity not supported by exchange (binance)")
        logging.debug("Trading granularity : {}".format(self.period_table[str(self.trading_granularity)]))
        logging.debug("Trading period : {}m".format(int(self._period/60)))
        # to be determined later
        #self.inputs = np.zeros([self.feature_number, self.coin_number, self.window])  # to be determined later
        self.inputs = np.zeros([self.feature_number, self.coin_number, self.window]) #self.get_first_inputs() #keep last
        #self.last_actual_time_slot = self.last_actual_time_slot - self._period

    # def test_pv(self):
    #     return self.__test_pv
    #
    # @property
    # def test_pc_vector(self):
    #     return np.array(self.__test_pc_vector, dtype=np.float32)
    #
    # def finish_trading(self):
    #     self.__test_pv = self._total_capital
    #
    #     """
    #     fig, ax = plt.subplots()
    #     ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
    #            self._rolling_trainer.data_matrices.sample_count)
    #     fig.tight_layout()
    #     plt.show()
    #     """
    #
    # def _log_trading_info(self, time, omega):
    #     pass
    #
    # def _initialize_data_base(self):
    #     pass
    #
    # def _write_into_database(self):
    #     pass
    #
    # # def __get_matrix_X(self):
    # #     return self.__test_set["X"][self._steps]
    # #
    # # def __get_matrix_y(self):
    # #     return self.__test_set["y"][self._steps, 0, :]
    #
    # # def __get_bought_y(self,input):
    # #     # price change from last buying price to current buying price
    # #     # approximated by market price for simulations
    # #     # print("input[0,:,-1]", input[0, :, -1])
    # #     # print("input[0,:,-2]", input[0, :, -2])
    # #     # print("input[:,:,-2:]", input[:, :, -2:])
    # #     self.__y = input[0,:,-1] / input[0,:,-2]

    # def rolling_train(self, online_sample=None):
    #     self._rolling_trainer.rolling_train()


    def __y(self):
        self.prices[1:] = self.inputs[0, :, -1]
        return self.prices[1:] / self.inputs[0,:,-2]

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        # logging.debug("the raw omega is {}".format(omega))
        ### POST Neural Network
        ####################################
        # portfolio value with commission deduction at the beginning of last period
        pv_after_last_commission = calculate_pv_after_commission(self._last_omega, self._last_omega_before_commission,
                                                                 self._commission_rate)  # for this period
        ####################################
        # get updated prices
        self.current_price[1:] = self.__y()
        ####################################
        # increase or decrease of portfolio value during last period
        portfolio_change = pv_after_last_commission * np.dot(self._last_omega, self.current_price)
        self._total_capital *= portfolio_change
        ####################################
        # calculate last_omega change until before this period
        self._last_omega_before_commission = pv_after_last_commission * self._last_omega * \
                                             self.current_price / \
                                             portfolio_change
        ####################################
        # calculate portfolio value for the beginning of next period
        self.pv_after_next_commission = calculate_pv_after_commission(omega, self._last_omega_before_commission,
                                                                      self._commission_rate)  # used to calculate last_omega_after_change
        self.__test_pc_vector.append(portfolio_change)
        ####################################
        # log results
        coins = np.concatenate((['BTC'],self._coin_name_list))
        self.prices[1:] = self.inputs[0, :, -1]
        summary_period = np.array(list(zip(coins, self.current_price, self._last_omega_before_commission)))
        predictions = np.array(list(zip(coins, omega)))
        self.qtties = self.__qtties(self.prices, self._last_omega_before_commission, self._total_capital)
        qtties_to_trade = self.qtties_to_trade(omega)
        # logging.debug("Prices : {}".format(list(zip(coins, self.prices))))
        logging.debug("Price changes, Max : {}, Min : {}, UP : {}, DOWN : {}, STABLE : {}".format(
            tuple(summary_period[np.argmax(self.current_price)].tolist()),
            tuple(summary_period[np.argmin(self.current_price)].tolist()),
            len(self.current_price[self.current_price > 1.0]),
            len(self.current_price[self.current_price < 1.0]), len(self.current_price[self.current_price == 1.0])))
        logging.debug(
            "Chosen last period : {}".format([tuple(a) for a in summary_period[self._last_omega_before_commission > 0].tolist()]))
        logging.debug(
            "Ownership : {}".format([tuple(a) for a in np.array(list(zip(coins, self.qtties)))[self.qtties > 0]]))
        logging.debug("Predictions : {}".format([tuple(a) for a in predictions[omega > 0].tolist()]))
        logging.debug("Execute : {}".format(
            [tuple(a) for a in np.array(list(zip(coins, qtties_to_trade)))[qtties_to_trade != 0]]))
        logging.debug("The portfolio change during last period is : {}".format(portfolio_change))
        logging.info('full portfolio change : {}'.format(self._total_capital / self._initial_btc))
        logging.info('total assets are {} BTC ({} USD)'.format(self._total_capital,self._total_capital * self.current_bitcoin_price))
        logging.debug("=" * 30 + "(SIMULATION)")
        ####################################

    def call_api(self, url):
        try:
            ret = urlopen(Request(url))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        except:
            pass

    def round_omega(self,omega):
        # ########################
        # # Experiment increase last prices by slippage
        #
        # self.inputs[0,:,-1] /= 0.55
        #
        # ########################
        if self.current_bitcoin_price > 0.0:
            price_BTC = self.current_bitcoin_price
        else:
            price_BTC = 10000
        limit_usd_in_BTC = self.min_usd_trade / price_BTC
        self.threshold = limit_usd_in_BTC / self._total_capital # an approximation on total capital from last period is ok here because fetching total capital before execution takes time
        zero_indices = []
        for i,w in enumerate(omega):
            if w <= self.threshold:
                zero_indices.append(i)
        len_zero_indices = len(zero_indices)
        len_omega = len(omega)
        if len_zero_indices == len_omega:
            omega = np.concatenate((np.ones(1), np.zeros(len_omega - 1)))
        else:
            omega = np.array(omega)
            omega += sum(omega[zero_indices]) / (len_omega - len_zero_indices)
            omega[zero_indices] = 0
            if sum(omega) != 0:
                omega = omega / sum(omega)
            else:
                omega = np.concatenate((np.ones(1), np.zeros(len_omega - 1)))
        return omega

    def last_omega_after_change(self):
        ### PRE Neural Network
        ####################################
        # get updated prices
        #self.current_price[1:] = self.__y()
        ####################################
        # get portfolio change for the latest period
        portfolio_change = self.pv_after_next_commission * np.dot(self._last_omega, self.current_price)
        ####################################
        # return the latest updated omega before insertion into the net
        return self.pv_after_next_commission * self._last_omega * self.current_price / portfolio_change

    def _last_omega_after_change(self,omega_before_change):
        return self.current_price * omega_before_change * self._commission_ratio()

    def __qtties(self, prices, omega, total_cap):
        return total_cap * omega / prices

    def qtties_to_trade(self,omega):
        weights_change = omega - self._last_omega_before_commission
        #sum_unchanged_weights = sum(weights_change[weights_change==0])
        spared_from_commission = sum(omega[weights_change<=0])
        minimum_total_capital_after_commission = self._total_capital * ( 1 - spared_from_commission ) * ( 1 - 2 * self._commission_rate ) # substract from total_capital the maximum possible commission
        self.next_qtties = self.__qtties(self.prices, omega, minimum_total_capital_after_commission)
        ###############################
        # unchanged_qtties_diff = (self.next_qtties-self.qtties)[weights_change==0]
        # unchanged_BTC_diff = sum(unchanged_qtties_diff * self.prices[weights_change==0])
        # if len(self.next_qtties[weights_change!=0]) != 0:
        #     self.next_qtties[weights_change!=0] = self.next_qtties[weights_change!=0] - unchanged_BTC_diff/len(self.next_qtties[weights_change!=0]) * self.prices[weights_change!=0]
        ###############################
        self.next_qtties[weights_change == 0] = self.qtties[weights_change == 0]
        return self.next_qtties - self.qtties

    def get_granular_data(self,pair,period,start,window):
        pass

    def generate_history_matrix(self): # checked only for binance
        last_input = self.get_last_input()
        if self.out_of_sync:
            self.inputs = last_input
            self.out_of_sync = False
            #logging.debug("Refetching succesful...")
        else:
            self.inputs = np.concatenate((self.inputs[:,:,1:] , last_input),axis=2)
        # logging.debug("Last fetched: {}".format(
        #     datetime.fromtimestamp(self.last_actual_time_slot+7200).strftime('%Y-%m-%d %H:%M')))
        # logging.debug("{}".format(
        #     datetime.fromtimestamp(time.time()+7200).strftime('%Y-%m-%d %H:%M:%S')))
        # logging.info("Trading every {} minutes on minute {}".format(int(self._period / 60),
        #                                                             int(self.trading_delay / 60)))
        # logging.debug("The portfolio change during last period is : {}".format(self.change))
        # ########################
        # # Experiment increase last prices by slippage
        #
        # self.inputs[0,:,-1] *= 0.55
        #
        # ########################
        #self._last_omega = self._last_omega_after_change(self._last_omega) # will be sent to NN now
        return self.inputs

    def get_first_inputs(self,when):
        actual_time = int(when)
        actual_time_slot = actual_time - actual_time % self.trading_granularity
        # logging.debug("Fetching full data matrix...")
        res = self._candelsticks(self.get_inputs(actual_time_slot, int(self.window * self._period / self.trading_granularity), self.trading_granularity))
        self.last_actual_time_slot = actual_time_slot
        return res

    def get_last_input(self):
        actual_time = int(time.time())
        actual_time_slot = actual_time - actual_time % self.trading_granularity
        if self.last_actual_time_slot == actual_time_slot - self._period:
            inputs = self.get_inputs(actual_time_slot, int(1 * self._period / self.trading_granularity), self.trading_granularity)
            res =  self._candelsticks(inputs)
            self.latest_commission_coin_price = self.fetch_coin_price(self.COMMISSION_COIN)  # get BNB price at about the same time as the other coins
            self.last_actual_time_slot = actual_time_slot
            return res
        # elif self.last_actual_time_slot == actual_time_slot + self.trading_granularity - self._period: # case when we arrive 1 granularity point before the period
        #     self.penultimate = self.get_inputs(actual_time_slot, int(1 * self._period / self.trading_granularity - self.trading_granularity),self.trading_granularity)
        #     now = time.time()
        #     time.sleep(now - now % self.trading_granularity)
        #     ultimate = self.get_last_tickers()
        #     res = self._candelsticks(np.concatenate((self.penultimate, ultimate),axis=2))
        #     self.last_actual_time_slot = actual_time_slot + self.trading_granularity
        #     return res
        else:
            self.out_of_sync = True
            #logging.debug("input matrix out of sync, refetching...")
            now = int(time.time())
            when = int(now - now % self._period + self.trading_delay)
            return self.get_first_inputs(when)

    def get_inputs(self,actual_time_slot, window, period):
        coins = self._coin_name_list
        # print("Selected coins:",coins)
        # server_time = int(self.call_api('https://api.binance.com/api/v1/time')['serverTime'] / 1000)  # binance
        # server_time_slot = server_time - server_time % period
        # print("server time slot:", server_time_slot, "current time slot:", actual_time_slot, datetime.fromtimestamp(actual_time_slot + 7200).strftime('%Y-%m-%d %H:%M'))
        # if server_time_slot != actual_time_slot:
        #     print("Server time and actual time slots are different...")
        #     print("server period:", server_time_slot, "current period:", actual_time_slot)
        #     time.sleep(5)
        #     raise Exception
        table = []
        start = (actual_time_slot - window * period) * 1000
        logging.debug("Fetching candlesticks from time {} to {}. Total {}.".format(
            datetime.fromtimestamp(start / 1000 + 7200).strftime('%Y-%m-%d %H:%M'),
            datetime.fromtimestamp(actual_time_slot - period + 7200).strftime('%Y-%m-%d %H:%M'), window))
        # for i in range(len(coins)):
        #     ohlcv = self.fetch_coin_inputs(coins[i],actual_time_slot,window,period)
        #     table.append(ohlcv)
        table = self.fetch_all_coins_in_parallel(coins,actual_time_slot,window,period)
        self.fetching_time = time.time() - self.start
        t = np.array(table)
        t = t.T
        if self.feature_number == 5:
            inputs = np.array([t[4].T, t[2].T, t[3].T, t[1].T, t[5].T])
        elif self.feature_number == 4:
            inputs = np.array([t[4].T, t[2].T, t[3].T, t[1].T])
        elif self.feature_number == 3:
            inputs = np.array([t[4].T, t[2].T, t[3].T])
        elif self.feature_number == 2:
            inputs = np.array([t[4].T, t[5].T])
        elif self.feature_number == 1:
            inputs = np.array([t[4].T])
        else:
            raise Exception("Number of features not supported...")
        return inputs

    def fetch_all_coins_in_parallel(self, coins, actual_time_slot, window, period):
        #############
        table = [[]] * len(coins)
        threads = []
        ##
        def fetch_coin_inputs_parallel(i, coin, actual_time_slot, window, period):
            table[i] = self.fetch_coin_inputs(coin, actual_time_slot, window, period)
        #############
        for i, coin in enumerate(coins):
            t = Thread(target=fetch_coin_inputs_parallel, args=(i, coin, actual_time_slot, window, period))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        return table


    def fetch_coin_inputs(self,coin,actual_time_slot,window,period):
        if coin == "reversed_USDT":
            pair = "BTC/USDT"
        else:
            pair = coin + "/BTC"
        limit = window
        start = (actual_time_slot - window * period) * 1000
        ohlcv = []
        while len(ohlcv) != window:
            fetch_ohlcv = self.exchange.fetch_ohlcv(pair, self.period_table[str(period)], start, limit)
            last_date = fetch_ohlcv[-1][0]
            # if self._exchange != "binance":
            #     time.sleep(10)
            if last_date > (actual_time_slot - period) * 1000:
                raise Exception("Problem fetching inputs... size is not right...Maybe due to missing data from server")
            if last_date == (actual_time_slot - period) * 1000:
                ohlcv.extend(fetch_ohlcv)
            if last_date < (actual_time_slot - period) * 1000:
                ohlcv.extend(fetch_ohlcv)
                start = last_date + (period * 1000)
                limit = max(1, window - len(ohlcv))
        if pair == "BTC/USDT":
            self.current_bitcoin_price = ohlcv[-1][4]
            ohlcv = np.array(ohlcv)
            ohlcv[:, -1] = ohlcv[:, -1] * ohlcv[:, 4]  # volume
            ohlcv[:, 1] = (1.0 / ohlcv[:, 1])  # open
            ohlcv[:, 4] = (1.0 / ohlcv[:, 4])  # close
            inverse_high = (1.0 / ohlcv[:, 2])  # high = 1/high
            inverse_low = (1.0 / ohlcv[:, 3])  # low = 1/low
            ohlcv[:, 2] = inverse_low  # low = 1/high
            ohlcv[:, 3] = inverse_high  # high = 1/low
            ohlcv = ohlcv.tolist()
        return ohlcv

    def _candelsticks(self,inputs):
        if self.trading_granularity == self._period:
            return inputs
        else:
            period = int(self._period / self.trading_granularity)
            num_features = inputs.shape[0]
            num_coins = inputs.shape[1]
            num_tickers = inputs.shape[2]
            #check integrity
            if num_tickers % period != 0 :
                raise Exception("Problem with granularity of input matrix... should be a divider of trading period and a multiple of 60")
            candlesticks = np.zeros([num_features, num_coins, int(num_tickers/period)])
            for i in range(0,num_tickers,period):
                if i % period == 0:
                    ticker = int(i / period)
                    candlesticks[0,:,ticker] = inputs[0,:,i+period-1]
                    if num_features >= 3:
                        candlesticks[1, :, ticker] = np.amax(inputs[1, :, i:i + period], axis=1)
                        candlesticks[2, :, ticker] = np.amin(inputs[2, :, i:i + period], axis=1)
                    if num_features == 4:
                        candlesticks[3, :, ticker] = inputs[3, :, i]
                    if num_features == 2 or num_features == 5:
                        candlesticks[4, :, ticker] = np.sum(inputs[4, :, i:i + period], axis=1)
            return candlesticks

    def get_last_tickers(self):
        prices = self.call_api('https://api.binance.com/api/v3/ticker/price')  # all prices from binance
        coins = np.concatenate((self.coins[1:], [self.COMMISSION_COIN]))
        coins[coins=="reversed_USDT"] = "USDT"
        ultimate = self.make_ultimate(coins,prices) # this ultimate includes commission coin
        self.latest_commission_coin_price = ultimate[0, -1, 0] # get BNB price at about the same time as the other coins
        return ultimate[:, :-1, :]

    def make_ultimate(self,coins,prices):
        ultimate = []
        for coin in coins:
            if coin == "USDT":
                symbol = 'BTCUSDT'
            else:
                symbol = coin + 'BTC'
            for pair in prices:
                if symbol == pair['symbol']:
                    price = float(pair['price'])
                    if symbol == 'BTCUSD' and price !=0 :
                        price = 1.0 / price
                    ultimate.append([price])
        if self.feature_number == 5:
            return np.array([ ultimate , ultimate , ultimate , ultimate , self.penultimate[4,:,-1] ])
        elif self.feature_number == 4:
            return np.array([ ultimate , ultimate , ultimate , self.penultimate[0,:,-1] ])
        elif self.feature_number == 3:
            return np.array([ ultimate , ultimate , ultimate ])
        elif self.feature_number == 2:
            return np.array([ ultimate , ultimate ])
        elif self.feature_number == 1:
            return np.array([ ultimate ])
        else:
            raise Exception("Number of features not supported...")

    def omega_update_before_prediction(self):
        self.current_price[1:] = self.__y()
        if self.simulation:
            return self.last_omega_after_change()
        else:
            self.fetch_updates_from_exchange()
            self._total_capital_before_slippage = self._total_capital
            if self.first_time:
                self.initialize_variables()
                return self.get_first_omega()
            else:
                return self._last_omega_after_change(self._last_omega)

    def execute_trade(self,omega):
        if not self.simulation:
            usdt_transaction = False
            buy_info = []
            buy = []
            for i,weight in enumerate(omega):
                coin = self.coins[i]
                owned_qtty = self.qtties[i]
                predicted_qtty = self._total_capital * weight / self.prices[i]
                transaction_qtty = predicted_qtty - owned_qtty
                (c, q) = self.correct_quote(coin, transaction_qtty)
                if coin != "BTC": #c is a pair, coin is a coin
                    if q < 0 :
                        if coin =="reversed_USDT": # needs to sell BTC after selling other coins otherwise there are no BTC available
                            usdt_i = i
                            usdt_c = c
                            usdt_q = q
                            usdt_transaction = True
                        else:
                            if self.fullfills_minimum_qtty(i,q):
                                self.sell_til_success(c, -q) #sell first
                    if q > 0 :
                        if coin == "reversed_USDT":
                            if self.fullfills_minimum_qtty(i, q):
                                self.buy_last_btc_usdt_til_success(c,q) # buying BTC on BTC/USDT is like selling USDT therefore we sell USDT
                        else:
                            weight_to_buy = transaction_qtty * self.prices[i] / self._total_capital
                            buy_info.append((i,weight_to_buy,q)) # not adding BTC/USDT buy (already done!)
                            #buy.append((i,c,q))
            ########################
            if usdt_transaction:
                if self.fullfills_minimum_qtty(usdt_i, usdt_q):
                    self.sell_last_btc_usdt_til_success(usdt_c, abs(usdt_q))  # sell BTC to get USDT
            ########################
            self.fetch_qtties_from_exchange() # needed to get the new BTC value (available capital)
            logging.debug("Total capital before selling: {}".format(self._total_capital))
            logging.debug("Slippage after selling: {}".format(self._total_capital/np.dot(self.prices, self.qtties)))
            available_capital = self.qtties[0] #BTC value
            total_buying_weight = 0
            for _,weight,_ in buy_info:
                total_buying_weight += weight
            logging.debug("Total buying amount before selling: {}".format(self._total_capital*total_buying_weight))
            logging.debug("Available buying amound: {} ({}% of initial)".format(available_capital,100*available_capital/(self._total_capital*total_buying_weight)))
            for coin_num,weight,qt in buy_info:
                coin = self.coins[coin_num]
                new_weight = weight / total_buying_weight
                qtty = available_capital * new_weight / self.prices[coin_num]
                if self.fullfills_minimum_qtty(coin_num, qtty):
                    (c,q) = self.correct_quote(coin, qtty)
                    self.buy_til_success(c,abs(q))  # then buy
                    logging.debug("original qtty: {}".format(qt))
            ##################
            # for i,c,q in buy:
            #     if self.fullfills_minimum_qtty(i, q):
            #         self.buy_til_success(c,0.99*abs(q))  # then buy



            # if len(buy_info) > 0:
            #     coin_num, weight = buy_info[-1]
            #     self.buy_last_til_success(coin_num,weight)


    def buy_til_success(self,coin,qtty):
        completed = False
        step = 0
        max_step = 500
        res = {}
        while not completed:
            logging.debug("Buying {} {}".format(qtty,coin))
            try:
                res = self.exchange.create_market_buy_order(coin, qtty)
            except:
                pass
            if res:
                if res['info']['status'] == 'FILLED':
                    logging.debug("Successful")
                    logging.debug("amount bought: {}".format(res["amount"]))
                    #print(res)
                    # rounding = qtty - res['amount']
                    # if  rounding > 0 :
                    #     self.__update_rounding(rounding,coin)
                    completed = True
                    return completed
            else:
                #completed = True
                logging.debug("Failed to BUY.\ncoin {}\nqtty {}".format(str(coin),str(qtty)))
                step += 1
                qtty *= 0.95  # retry with a lower qtty
                logging.debug("{}/{}: Retrying  with lower quantity {}".format(step, max_step, qtty))
                completed = False
                if step == max_step:
                    completed = True
                    self.fetch_qtties_from_exchange()
                    print("Couldn't buy asset {}".format(coin))
                    print(self.balance)
                    # print("Currently owned : {} {}".format(self.qtties[coin_num], self.coins[coin_num]))
                    # print("Updated weight :", weight)
                    print("Total BTC available : {} BTC".format(self.qtties[0]))

    def sell_til_success(self, coin, qtty):
        completed = False
        step=0
        max_step=500
        res = {}
        while not completed:
            logging.debug("Selling {} {}".format(qtty, coin))
            try:
                res = self.exchange.create_market_sell_order(coin, qtty)
            except:
                pass
            if res:
                if res['info']['status'] == 'FILLED':
                    logging.debug("Successful")
                    logging.debug("amount sold: {}".format(res["amount"]))
                    # print(res)
                    # rounding = qtty - res['amount']
                    # if rounding > 0:
                    #     self.__update_rounding(rounding, coin)
                    completed = True
                    return completed
            else:
                #completed = True
                logging.debug("Failed to SELL.\ncoin {}\nqtty {}".format(str(coin),str(qtty)))
                step += 1
                qtty *= 0.95  # retry with a lower qtty
                logging.debug("{}/{}: Retrying  with lower quantity {}".format(step, max_step, qtty))
                completed = False
                if step == max_step:
                    completed = True
                    self.fetch_qtties_from_exchange()
                    print("Couldn't sell asset {}".format(coin))
                    print(self.balance)
                    #print("Currently owned : {} {}".format(self.qtties[coin_num], self.coins[coin_num]))
                    #print("Updated weight :", weight)
                    print("Total BTC available : {} BTC".format(self.qtties[0]))

    def update_after_trade(self):
        pass

    def print_changes_after_fetching(self):
        pass

    def print_changes_after_prediction(self, omega):
        predictions = np.array(list(zip(self.coins, omega)))
        logging.debug("Current BTC price : {} USD".format(self.current_bitcoin_price))
        logging.debug("Minimum USD trade is {} USD. Percentage of portfolio : {}".format(self.min_usd_trade, self.threshold))
        logging.debug("=" * 30)
        logging.debug("Predicted : {}".format([tuple(a) for a in predictions[omega > 0].tolist()]))


    def print_changes_after_execution(self,predicted_omega):
        self.fetch_updates_from_exchange() #needed to get last bought coins qtty
        #############
        ###########################
        if self.silent:
            self.logger.setLevel(logging.DEBUG)
        ###########################
        self.slippage = self._total_capital_before_slippage / self._total_capital
        self.total_average_slippage = self.__update_average_slippage()
        self.total_accumulated_slippage *= self.slippage
        self.total_capital_with_commission = self._total_capital * self._commission_ratio()
        self.full_exchange_capital = self._total_capital + self.qtty_commission_coin * self.latest_commission_coin_price + self.unused_usdt / self.current_bitcoin_price + self.unused_btc
        self.change = self.total_capital_with_commission / self.last_total_capital_with_commission
        self.total_change *= self.change
        omega = predicted_omega
        # self.__rounding = self.__rounding / self.total_capital_with_commission
        # self._average_rounding = (self._average_rounding * self._steps + self.__rounding) / (self._steps + 1)
        ####################################
        # log results
        coins = self.coins.copy()
        coins[coins == "reversed_USDT"] = "USDT"
        #self.current_price[1:] = self.__y()
        #self.prices[1:] = self.inputs[0, :, -1]
        summary_period = np.array(list(zip(self.coins, self.current_price)))
        owned = np.array(list(zip(coins, self.qtties)))
        #transactions_sent = self.transactions.copy()
        predictions = np.array(list(zip(self.coins, omega)))
        executed_vc = self.qtties - self.last_qtties
        #not_executed_vc = self.transactions[:,1].astype(np.float) - executed_vc
        #not_executed = np.array(list(zip(coins, not_executed_vc)))
        executed = np.array(list(zip(coins, executed_vc)))
        logging.debug("=" * 30)
        logging.info("the step is {}".format(self._steps))
        # logging.debug("Prices : {}".format(list(zip(coins, self.prices))))
        logging.debug("-" * 30)
        logging.debug("Price changes, Best : {}, Worst : {}, Mean : {}, Median {}, UP : {}, DOWN : {}, STABLE : {}".format(
            tuple(summary_period[np.argmax(self.current_price)].tolist()),
            tuple(summary_period[np.argmin(self.current_price)].tolist()),
            np.mean(self.current_price),
            np.median(self.current_price),
            len(self.current_price[self.current_price > 1.0]),
            len(self.current_price[self.current_price < 1.0]), len(self.current_price[self.current_price == 1.0])))
        self.average_market_change *= np.mean(self.current_price)
        self.best_model_change *= np.max(self.current_price)
        # logging.debug(
        #     "Chosen last period : {}".format(
        #         [tuple(a) for a in summary_period[self._last_omega > 0].tolist()]))
        logging.debug("=" * 30)
        # logging.debug("Rounding this period: {} (av.: {})".format(self.__rounding, self._average_rounding))
        logging.debug("Slippage this period : {}".format(self.slippage))
        logging.debug("Total accumulated slippage : {}".format(self.total_accumulated_slippage))
        logging.debug("Average slippage : (geometric) {}, (arithmetic) {}".format(self.total_accumulated_slippage**(1/(self._steps+1)), self.total_average_slippage))
        logging.debug("-" * 30)
        logging.debug("Best possible model : {} (av.: {}) - wo slippage : {} (av.: {})".format(self.best_model_change * self._commission_ratio() / self.total_accumulated_slippage, (self.best_model_change * self._commission_ratio() / self.total_accumulated_slippage)**(1/(self._steps+1)), self.best_model_change * self._commission_ratio(), (self.best_model_change * self._commission_ratio())**(1/(self._steps+1))))
        logging.debug("Average market model : {} (av.: {}) - wo slippage : {} (av.: {})".format(self.average_market_change * self._commission_ratio() / self.total_accumulated_slippage, (self.average_market_change * self._commission_ratio() / self.total_accumulated_slippage)**(1/(self._steps+1)), self.average_market_change * self._commission_ratio(), (self.average_market_change * self._commission_ratio())**(1/(self._steps+1))))
        logging.debug("Current model : {} (av.: {}) - wo slippage {} (av.: {})".format(self.total_change, self.total_change ** (1 / (self._steps + 1)), self.total_change * self.total_accumulated_slippage, (self.total_accumulated_slippage * self.total_change) ** (1 / (self._steps + 1))))
        logging.debug("=" * 30)
        logging.debug("Predicted : {}".format([tuple(a) for a in predictions[omega > 0].tolist()]))
        #logging.debug("Transactions sent : {}".format([tuple(a) for a in transactions_sent[self.fullfills_minimum_qtties(transactions_sent[:,1].astype(np.float))].tolist()]))
        logging.debug("Executed : {}".format([tuple(a) for a in executed[self.fullfills_minimum_qtties(executed_vc)].tolist()]))
        #logging.debug("Not executed : {}".format([tuple(a) for a in not_executed[self.fullfills_minimum_qtties(not_executed_vc)].tolist()]))
        logging.debug("Ownership : {}".format([tuple(a) for a in owned[self.fullfills_minimum_qtties(self.qtties)]]))
        #logging.debug("The portfolio change during last period is : {}".format(change))
        logging.debug("=" * 30)
        logging.debug("Model change this period : {} - wo slippage {}".format(self.change, self.change * self.slippage))
        logging.debug("-" * 30)
        logging.info('Initial capital : {} BTC of {} BTC / {} USD of {} USD on {}'.format(self.initial_capital, self.initial_full_exchange_capital, self.initial_usd_capital, self.initial_usd_full_exchange_capital, datetime.fromtimestamp(self.started_trading + 7200).strftime('%Y-%m-%d %H:%M')))
        logging.info('Current capital : {} BTC of {} BTC / {} USD of {} USD on {}'.format(self.total_capital_with_commission, self.full_exchange_capital, self.total_capital_with_commission * self.current_bitcoin_price, self.full_exchange_capital * self.current_bitcoin_price, datetime.fromtimestamp(time.time() + 7200).strftime('%Y-%m-%d %H:%M')))
        logging.debug("=" * 30)
        self.last_qtties = self.qtties.copy()
        self.last_qtty_commission_coin = self.qtty_commission_coin
        self.last_total_capital = self._total_capital
        self.last_total_capital_with_commission = self.total_capital_with_commission
        ####### rounding reset
        # self.__rounding = 0.0
        ####################################
        if self._total_capital < self.initial_capital * 0.8:
            raise Exception("Capital is being wasted. Lost 20%.\nStopping algorithm")
        ####################################

    def fetch_updates_from_exchange(self):
        self.fetch_qtties_from_exchange()
        # self.portfolio_btc_values = (self.prices * self.qtties) #- (commission_spent / len(self.portfolio_btc_values))
        # self._total_capital = sum(self.portfolio_btc_values) #- self.unused_btc - (self.unused_usdt /self.current_bitcoin_price)
        # we want full total capital including commission coin
        self._total_capital = sum(self.prices * self.qtties)

    def _commission_ratio(self):
        commission = (self.last_qtty_commission_coin - self.qtty_commission_coin) * self.latest_commission_coin_price
        return (self._total_capital - commission) / self._total_capital

    def fetch_qtties_from_exchange(self):
        self.balance = self.exchange.fetch_balance()
        ########### get BNB qtty
        self.qtty_commission_coin = self.balance['free'][self.COMMISSION_COIN]
        #### get quantities
        qtt = []
        for coin in self.coins:
            coin = self.correct_name(coin)
            if coin == "BTC":
                qtt.append(self.balance['free'][coin] - self.unused_btc)
            elif coin == "USDT":
                qtt.append(self.balance['free'][coin] - self.unused_usdt)
            else:
                qtt.append(self.balance['free'][coin])
        self.qtties = np.array(qtt)
        ####

    def initialize_variables(self):
        self.last_qtties = self.qtties.copy()
        self.last_total_capital = self._total_capital
        self.initial_capital = self._total_capital
        self.initial_usd_capital = self.initial_capital * self.current_bitcoin_price
        self.initial_full_exchange_capital = self.initial_capital + self.qtty_commission_coin * self.latest_commission_coin_price + self.unused_btc + self.unused_usdt / self.current_bitcoin_price
        self.initial_usd_full_exchange_capital = self.initial_full_exchange_capital * self.current_bitcoin_price
        self.last_total_capital_with_commission = self._total_capital
        ##########
        # qtt = []
        # for coin in self.coins:
        #     coin = self.correct_name(coin)
        #     if coin == "BTC":
        #         qtt.append(self.balance['free'][coin] - self.unused_btc)
        #     elif coin == "USDT":
        #         qtt.append(self.balance['free'][coin] - self.unused_usdt)
        #     else:
        #         qtt.append(self.balance['free'][coin])
        # self.qtties = np.array(qtt)
        ############
        self.first_time = False
        ############

    def correct_name(self,c):
        if c == "reversed_USDT":
            return "USDT"
        else:
            return c

    def correct_quote(self,c, q):
        if c == "reversed_USDT":
            return ("BTC/USDT", -q / self.current_bitcoin_price)
        else:
            return ((c + "/BTC"), q)

    def fullfills_minimum_qtty(self,i, q):
        MINIMUM_BTC = 0.001
        if self.coins[i] == "reversed_USDT":
            q = -q * self.current_bitcoin_price
        btc_equivalent = self.prices[i]*abs(q)
        if btc_equivalent >= MINIMUM_BTC:
            return True
        else:
            return False

    def fullfills_minimum_qtties(self, q):
        MINIMUM_BTC = 0.001
        btc_equivalent = self.prices * abs(q)
        return btc_equivalent >= MINIMUM_BTC

    # def buy_last_til_success(self, coin_num, weight):
    #     self.fetch_qtties_from_exchange()
    #     coin = self.coins[coin_num]
    #     qtty = self.qtties[0] / self.prices[coin_num]
    #     step = 0
    #     max_step = 1 # sell_til_success and buy_til_success already have loops
    #     keep_buying = True
    #     completed = False
    #     while keep_buying:
    #         if self.fullfills_minimum_qtty(coin_num, qtty):
    #             (c,q) = self.correct_quote(coin, qtty)
    #             try:
    #                 if coin == "reversed_USDT":
    #                     completed = self.sell_til_success(c,abs(q))
    #                 else:
    #                     completed = self.buy_til_success(c,abs(q))
    #             except:
    #                 pass
    #             keep_buying = not completed
    #         else:
    #             keep_buying = False
    #         if keep_buying:
    #             step += 1
    #             qtty *= 0.999  # retry with a lower qtty
    #             # print("{}/{}: Retrying  with lower quantity {}".format(step,max_step, qtty))
    #             if step == max_step:
    #                 keep_buying = False
    #                 self.fetch_qtties_from_exchange()
    #                 print("Couldn't buy last asset {}".format(self.correct_name(coin)))
    #                 print(self.balance)
    #                 print("Currently owned : {} {}".format(self.qtties[coin_num], self.correct_name(coin)))
    #                 print("Updated weight :", weight)
    #                 print("Total BTC available : {} BTC".format(self.qtties[0]))

    def buy_last_btc_usdt_til_success(self,coin,qtty):
        # coin = "BTC/USDT and qtty is positive in BTC
        completed = False
        step = 0
        max_step = 500
        res = {}
        while not completed:
            logging.debug("Selling {} USDT (buying {} BTC)".format(qtty*self.current_bitcoin_price,qtty))
            try:
                res = self.exchange.create_market_buy_order(coin, qtty)
            except:
                pass
            if res:
                if res['info']['status'] == 'FILLED':
                    logging.debug("Successful")
                    logging.debug("amount bought: {} BTC".format(res["amount"]))
                    # print(res)
                    # rounding = qtty - res['amount']
                    # if rounding > 0:
                    #     self.__update_rounding(rounding, coin)
                    completed = True
                    return completed
                else:
                    print("Not filled")

            if not completed:
                step += 1
                qtty *= 0.95  # retry with a lower qtty
                logging.debug("{}/{}: Retrying  with lower quantity {}".format(step, max_step, qtty))
                if step == max_step:
                    completed = True
                    self.fetch_qtties_from_exchange()
                    print("Couldn't sell USDT (buy BTC) on BTC/USDT")
                    print(self.balance)
                    print("Currently USDT available : {} USDT".format(self.balance["USDT"]["free"]))
                    print("Total BTC available : {} BTC".format(self.qtties[0]))

    def sell_last_btc_usdt_til_success(self, coin, qtty):
        # coin = "BTC/USDT and qtty is positive in BTC
        completed = False
        step = 0
        max_step = 500
        res = {}
        while not completed:
            logging.debug("Buying {} USDT (selling {} BTC)".format(qtty * self.current_bitcoin_price,qtty))
            try:
                res = self.exchange.create_market_sell_order(coin, qtty)
            except:
                pass
            if res:
                if res['info']['status'] == 'FILLED':
                    logging.debug("Successful")
                    logging.debug("amount sold: {} BTC".format(res["amount"]))
                    # print(res)
                    # rounding = qtty - res['amount']
                    # if rounding > 0:
                    #     self.__update_rounding(rounding, coin)
                    completed = True
                    return completed
                else:
                    print("Not filled")

            if not completed:
                step += 1
                qtty *= 0.95  # retry with a lower qtty
                logging.debug("{}/{}: Retrying  with lower quantity {}".format(step, max_step, qtty))
                if step == max_step:
                    completed = True
                    self.fetch_qtties_from_exchange()
                    print("Couldn't buy USDT (sell BTC) on BTC/USDT")
                    print(self.balance)
                    print("Currently USDT available : {} USDT".format(self.balance["USDT"]["free"]))
                    print("Total BTC available : {} BTC".format(self.qtties[0]))

    def fetch_coin_price(self,coin):
        if coin == 'USDT' or coin == 'reversed_USDT':
            pair = 'BTCUSDT'
        else:
            pair = coin + 'BTC'
        api_req = 'https://api.binance.com/api/v3/ticker/price?symbol=' + pair
        try:
            res = self.call_api(api_req)
            price = float(res['price'])
            if price:
                return price
        except:
            print("Coin " + coin + " can't be fetched..")
            if coin == self.COMMISSION_COIN:
                print("Returning last period price...")
                return self.latest_commission_coin_price
            else:
                return 0.0

    def get_first_omega(self):
        self.balance = self.exchange.fetch_balance()
        #self.prices[1:] = self.inputs[0, :, -1]
        #### get quantities
        qtt = []
        for coin in self.coins:
            coin = self.correct_name(coin)
            if coin == "BTC":
                qtt.append(self.balance['free'][coin] - self.unused_btc)
            elif coin == "USDT":
                qtt.append(self.balance['free'][coin] - self.unused_usdt)
            else:
                qtt.append(self.balance['free'][coin])
        self.qtties = np.array(qtt)
        ####
        self._total_capital = np.dot(self.qtties,self.prices)
        first_omega = self.qtties * self.prices / self._total_capital
        return self.round_omega(first_omega)

    def __update_average_slippage(self):
        return (self.total_average_slippage * self._steps + self.slippage) /(self._steps + 1)

    def __update_rounding(self,rounding,coin):
        price , c = self.__price_from_quote(coin)
        rounding_in_BTC = rounding * price
        # print("rounding of {} BTC , coin {}".format(rounding_in_BTC, c))
        self.__rounding += rounding_in_BTC

    def __price_from_quote(self,coin):
        c, _ = str(coin).split("/")
        ind = [i for i in range(len(self.coins)) if self.coins[i] == c][0]
        return self.prices[ind] , c

    def clean_portfolio(self):
        # sell all coins that are not in portfolio except BNB
        no_coins_to_sell = True
        logging.debug("Selling all non-traded coins if any...")
        MINIMUM_BTC = 0.001
        coins_on_exchange = np.array(list(self.balance["free"].keys()))
        coins_to_exclude = np.concatenate((self.coins, [self.COMMISSION_COIN]))
        coins_to_exclude[coins_to_exclude=="reversed_USDT"] = "USDT"
        coins_to_sell = [coin for coin in coins_on_exchange if coin not in coins_to_exclude]
        prices = self.call_api('https://api.binance.com/api/v3/ticker/price') # all prices
        if coins_to_sell:
            for coin in coins_to_sell:
                for pair in prices:
                    if coin + 'BTC' == pair['symbol']:
                        price = float(pair['price'])
                        qtty = self.balance['free'][coin]
                        btc_value = price * qtty
                        if btc_value >= MINIMUM_BTC:
                            (c,q) = self.correct_quote(coin,qtty)
                            self.sell_til_success(c,q)
                            no_coins_to_sell = False
        if no_coins_to_sell:
            print("no coins to sell...")

    def warm_up_connection(self):
        print("Warming up ...")
        self.exchange.fetch_balance()
        coins = self.coins[1:]
        for i in range(len(coins)):
            if coins[i] == "reversed_USDT":
                pair = "BTC/USDT"
            else:
                pair = coins[i] + "/BTC"
            self.exchange.fetch_ohlcv(pair, self.period_table[str(self.trading_granularity)], (int(time.time()) - self.window * self.trading_granularity) * 1000, self.window)
        self.exchange.fetch_balance()

    def easy_warm_up(self):
        self.exchange.fetch_balance()