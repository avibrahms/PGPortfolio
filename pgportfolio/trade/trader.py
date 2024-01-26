from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from pgportfolio.learn.rollingtrainer import RollingTrainer
import logging
import time


class Trader:
    def __init__(self, waiting_period, config, total_steps, net_dir, agent=None, initial_BTC=1.0, agent_type="nn"):
        """
        @:param agent_type: string, could be nn or traditional
        @:param agent: the traditional agent object, if the agent_type is traditional
        """
        self._steps = 0
        self._total_steps = total_steps
        self.config = config
        self._period = config["input"]["global_period"]
        if self._period % 60 != 0:
            raise Exception("Period shoulb be a multiple of 60")
        self._agent_type = agent_type
        if agent_type == "traditional":
            config["input"]["feature_number"] = 1
            config["input"]["norm_method"] = "relative"
            self._norm_method = "relative"
        elif agent_type == "nn":
            if self.__class__.__name__=="BackTest":
                config["input"]["is_backtest"] = True
            self._rolling_trainer = RollingTrainer(config, net_dir, agent=agent)
            self._coin_name_list = self._rolling_trainer.coin_list
            self.coins = np.concatenate((['BTC'], self._coin_name_list))
            self._norm_method = config["input"]["norm_method"]
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            if not agent:
                agent = self._rolling_trainer.agent
        else:
            raise ValueError()
        self._agent = agent

        # the total assets is calculated with BTC
        self.last_actual_time_slot = 0
        self._total_capital = config["input"]["initial_BTC"]
        self._initial_btc = config["input"]["initial_BTC"]
        self._window_size = config["input"]["window_size"]
        self._coin_number = config["input"]["coin_number"]
        self.silent = config["input"]["silent"]
        self.warmup = config["input"]["warmup"]
        self._commission_rate = config["trading"]["trading_consumption"]
        self._fake_ratio = config["input"]["fake_ratio"]
        self._asset_vector = np.zeros(self._coin_number+1)
        self.simulation = config["input"]["simulation"]
        self._last_omega = np.zeros((self._coin_number+1,))
        self._last_omega[0] = 1.0
        self.total_time = 1.0
        self.execution_time = 1.0
        self.prediction_time = 1.0
        self.fetch_data_time = 1.0
        self.fetching_time = 1.0
        self.start = 1.0
        print("Initial portfolio weights:",self._last_omega)

        if self.__class__.__name__=="BackTest":
            # self._initialize_logging_data_frame(initial_BTC)
            self._logging_data_frame = None
            # self._disk_engine =  sqlite3.connect('./database/back_time_trading_log.db')
            # self._initialize_data_base()
        self.trading_granularity = config["input"]["trading_granularity"]
        self.trading_delay = 60 * int(config["input"]["trading_delay"]) #// self.trading_granularity) * self.trading_granularity
        self._initial_delay = 120
        if self.trading_granularity == self._period:
            self.trading_delay = 0
        if self.trading_delay/60 * self.trading_granularity >= self._period:
            self.trading_delay = int(self.trading_delay % self._period)
            logging.debug("trading delay should be between {} and {}. Reajusted to {}".format(0,int(self._period/self.trading_granularity)-1,int(self.trading_delay/60)))

    def _initialize_logging_data_frame(self, initial_BTC):
        logging_dict = {'Total Asset (BTC)': initial_BTC, 'BTC': 1}
        for coin in self._coin_name_list:
            logging_dict[coin] = 0
        self._logging_data_frame = pd.DataFrame(logging_dict, index=pd.to_datetime([time.time()], unit='s'))

    def generate_history_matrix(self):
        """override this method to generate the input of agent
        """
        pass

    def finish_trading(self):
        pass

    # add trading data into the pandas data frame
    def _log_trading_info(self, time, omega):
        time_index = pd.to_datetime([time], unit='s')
        if self._steps > 0:
            logging_dict = {'Total Asset (BTC)': self._total_capital, 'BTC': omega[0, 0]}
            for i in range(len(self._coin_name_list)):
                logging_dict[self._coin_name_list[i]] = omega[0, i + 1]
            new_data_frame = pd.DataFrame(logging_dict, index=time_index)
            self._logging_data_frame = self._logging_data_frame.append(new_data_frame)

    def trade_by_strategy(self, omega):
        """execute the trading to the position, represented by the portfolio vector w
        """
        pass

    def rolling_train(self):
        """
        execute rolling train
        """
        pass

    def __trade_body(self):
        if self.__class__.__name__ == "RealTrade":
            self.start = int(time.time())
            history_matrix = self.generate_history_matrix() # to get latest prices
            last_omega = self.omega_update_before_prediction() # updates _last_omega and total_capital with real values from exchange
            fetch_data_time = time.time() - self.start
            omega = self._agent.decide_by_history(history_matrix,last_omega)
            omega = self.round_omega(omega)
            prediction_time = time.time() - fetch_data_time - self.start
            self.print_changes_after_prediction(omega)
            if self.simulation:
                self.trade_by_strategy(omega) # to be commented replaced by execute_trade
                self._last_omega = omega.copy()
            else:
                #time.sleep(10-time.time()%10)
                #after_stop = time.time()
                self.execute_trade(omega) # sell and buy assets according to predicted omega
                final = time.time()
                execution_time = final - prediction_time - fetch_data_time - self.start
                total_time = final - self.start
                self.print_changes_after_execution(omega) # equivalent of trade_by_stratey
                logging.info("Data fetching : {}s (av. {}s)\n- fetch : {}s\n- pre-process : {}s\nPrediction : {}s (av. {}s)\nExecution : {}s (av. {}s)\nTotal : {}s (av. {}s)".format(fetch_data_time, self.get_av(fetch_data_time,self.fetch_data_time), self.fetching_time, fetch_data_time - self.fetching_time, prediction_time,self.get_av(prediction_time,self.prediction_time),execution_time,self.get_av(execution_time,self.execution_time),total_time,self.get_av(total_time,self.total_time)))
                logging.debug("=" * 60)
                logging.debug("Total portfolio change : {}".format(self.total_change))
                self.total_time = self.get_av(total_time,self.total_time)
                self.execution_time = self.get_av(execution_time,self.execution_time)
                self.prediction_time = self.get_av(prediction_time,self.prediction_time)
                self.fetch_data_time = self.get_av(fetch_data_time,self.fetch_data_time)
                logging.debug("=" * 30)
                self._last_omega = omega.copy()
            sleep_time = self._period - (int(time.time()) % self.last_actual_time_slot)
            logging.info("sleep for %s seconds" % (sleep_time))
        else: # Backtest
            starttime = time.time()
            omega = self._agent.decide_by_history(self.generate_history_matrix(), self._last_omega.copy())
            omega = self._round_omega(omega,self._total_capital)
            self.trade_by_strategy(omega)
            trading_time = time.time() - starttime
            sleep_time = self._period - trading_time
        if self._agent_type == "nn":
            self.rolling_train()
        self._steps += 1
        return sleep_time

    def start_trading(self):
        try:
            if self.__class__.__name__=="RealTrade":
                now = time.time()
                self.started_trading = now
                if self.trading_delay < 0:
                    self.trading_delay = int((now - now % self.trading_granularity + max(self.trading_granularity, self._initial_delay)) % self._period)
                when = int(now - now % self._period + self.trading_delay - self._period)
                if when +self._period - now <=0:
                    when+=self._period
                if not self.simulation:
                    self.clean_portfolio()
                self.inputs = self.get_first_inputs(when)
                ######
                if self.warmup:
                    self.easy_warm_up()
                ######
                wait = self.trading_delay - time.time() % self._period
                if wait<=0:
                    wait+=self._period
                logging.info("Trading every {} minutes on minute {}".format(int(self._period / 60), int(self.trading_delay / 60)))
                logging.info("sleep for %s seconds" % int(wait))
                ###########################
                if self.silent:
                    self.logger.setLevel(logging.CRITICAL)
                ###########################
                ###################################
                # warm up if any
                if self.warmup and wait > 60:
                    time.sleep(wait - 60)
                    self.warm_up_connection()
                    small_wait = 60 - time.time() % 60
                    #print("sleep for %s seconds" % int(small_wait))
                    time.sleep(small_wait)
                else:
                    time.sleep(wait)


                while self._steps < self._total_steps:
                    sleeptime = self.__trade_body()
                    ###########################
                    if self.silent:
                        self.logger.setLevel(logging.CRITICAL)
                    ###########################
                    ###################################
                    # warm up if any
                    if self.warmup and sleeptime > 60:
                        time.sleep(sleeptime-60)
                        self.warm_up_connection()
                        small_sleep = 60 - time.time() % 60
                        #print("sleep for %s seconds" % int(small_sleep))
                        time.sleep(small_sleep)
                    else:
                        time.sleep(sleeptime)
                    ###################################
            else:
                ###########################
                if self.silent:
                    self.logger.setLevel(logging.CRITICAL)
                ###########################
                while self._steps < self._total_steps:
                    self.__trade_body()
                    if self._steps == self._total_steps:
                        return self._total_capital / self._initial_btc

        finally:
            if self._agent_type=="nn":
                self._agent.recycle()
            self.finish_trading()

    def round_omega(self,omega):
        pass

    def _round_omega(self, omega, total_capital):
        pass

    def last_omega_after_change(self):
        pass

    def omega_update_before_prediction(self):
        self.update()

    def update_after_trade(self):
        self.update()

    def update(self):
        pass

    def execute_trade(self,omega):
        pass

    def print_changes_after_fetching(self):
        pass

    def print_changes_after_prediction(self,o):
        pass

    def print_changes_after_execution(self,omega):
        pass

    def print_account_info(self,exchange):
        print("Account balance:\n",exchange.fetch_balance())

    def connect_api(self):
        if not self.simulation and self.__class__.__name__=="RealTrade":
            api_connected = False
            print("Are you sure you want to start trading? If yes.")
            while not api_connected:
                try:
                    ########################################
                    apikey = input("Type your API key:")
                    secretkey = input("Enter your secret key:")
                    self.exchange.apiKey = apikey
                    self.exchange.secret = secretkey
                    if self.exchange.apiKey == '1' and self.exchange.secret == '11':
                        self.exchange.apiKey = "o06CM6TnllHuYb4EtwlgXXiylQM676yBvyk0rjUsqYIFl6BKazpqdxmXDc39uGA5" #to erase
                        self.exchange.secret = "RwQl3rqyzM4G651Ykq05ncFAsc0iba4ggMDEd8tmtAVZum9cweLMa4bCdKQAMGlY" #to erase
                    if self.exchange.apiKey == '2' and self.exchange.secret == '22':
                        self.exchange.apiKey = "U441rSAPzE0WutiIlsLStYWBeCG0yltueB4zTI9h9ApVt1gJLEj0PgQIR9gKW8hK" #to erase
                        self.exchange.secret = "CnkSybUoaGphpiFOu9jFSLFncmq1UyOR36VvxTIKkwvib25qtJ6hW9rQswsq3tmK" #to erase
                    ########################################
                    #self.print_account_info(self.exchange)
                    api_connected = True
                    input("Press ENTER twice to start trading...")
                except:
                    print("API or/and secret key not valid. Try again:")

    def initialize_variables(self):
        pass

    def get_first_inputs(self,when):
        pass

    def clean_portfolio(self):
        pass

    def warm_up_connection(self):
        pass

    def easy_warm_up(self):
        pass

    def get_av(self,value,av_value):
        return ( av_value * self._steps + value ) / ( self._steps + 1 )

    def change_agent(self,algo,device="cpu"):
        # close session and graph
        self._agent.recycle()
        # change agent
        self._rolling_trainer.change_agent(algo,self._coin_number,device)
        self._agent = self._rolling_trainer._agent
        # reinitialize variables
        self._steps = 0
        self._last_omega = np.zeros((self._coin_number+1,))
        self._last_omega[0] = 1.0
        self._total_capital = self.config["input"]["initial_BTC"]
        self._initial_btc = self.config["input"]["initial_BTC"]
        self.__test_pv = 1.0
        self.__test_pc_vector = []


    def change_portfolio(self,coin_number):
        self._coin_number = coin_number
        return self._coin_number