from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission


class BackTest(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        elif agent_type == "traditional":
            config["input"]["feature_number"] = 1
            data_matrices = DataMatrices.create_from_config(config)
        else:
            raise ValueError()
        self.__test_set = data_matrices.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
        self._total_steps = self.__test_length
        self.__test_pv = 1.0
        self.__test_pc_vector = []

    @property
    def test_pv(self):
        return self.__test_pv

    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        return self.__test_set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__test_set["y"][self._steps, 0, :]

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        inputs = self.__get_matrix_X()
        inputs = inputs[:,:self._coin_number,:] #select coins for multi_backtesting / coin_number is changes everytime
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        matrix_y = self.__get_matrix_y()
        matrix_y = matrix_y[:self._coin_number] #select coins for multi_backtesting / coin_number is changes everytime
        future_price = np.concatenate((np.ones(1), matrix_y)) # for next period
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate) # for this period
        portfolio_change = pv_after_commission * np.dot(omega, future_price) #for next period
        self._total_capital *= portfolio_change #for next period
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change # for next period!
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
        logging.info('total assets are %3f BTC' % self._total_capital)
        logging.info('full portfolio change : {}'.format(self._total_capital / self._initial_btc))
        logging.debug("=" * 30)


    def _round_omega(self,omega,total_capital):
        price_BTC = 10000
        twenty_usd_in_BTC = 20.0 / price_BTC
        threshold = twenty_usd_in_BTC / total_capital
        zero_indices = []
        for i,w in enumerate(omega):
            if w <= threshold:
                zero_indices.append(i)
        len_zero_indices = len(zero_indices)
        len_omega = len(omega)
        if len_zero_indices == len_omega:
            omega = np.concatenate((np.ones(1),np.zeros(len_omega-1)))
        else:
            omega = np.array(omega)
            omega += sum(omega[zero_indices])/(len_omega-len_zero_indices)
            omega[zero_indices]=0
            if sum(omega) != 0:
                omega = omega / sum(omega)
            else:
                omega = np.concatenate((np.ones(1), np.zeros(len_omega - 1)))
        return omega
