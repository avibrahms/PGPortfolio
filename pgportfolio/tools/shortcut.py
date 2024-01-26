from __future__ import division,absolute_import,print_function
from pgportfolio.trade.backtest import BackTest
from pgportfolio.trade.realtrade import RealTrade
import pandas as pd
import os
import time
#from pgportfolio.trade.testtrade import TestTrade
#from pgportfolio.tdagent.algorithms import crp, ons, olmar, up, anticor1, pamr,\
    # best, bk, cwmr_std, eg, sp, ubah, wmamr, bcrp, cornk, m0, rmr


# the dictionary of name of algorithms mapping to the constructor of tdagents
ALGOS = {}#{"crp": crp.CRP, "ons": ons.ONS, "olmar": olmar.OLMAR, "up": up.UP,
         # "anticor": anticor1.ANTICOR1, "pamr": pamr.PAMR,
         # "best": best.BEST, "bk": bk.BK, "bcrp": bcrp.BCRP,
         # "corn": cornk.CORNK, "m0": m0.M0, "rmr": rmr.RMR,
         # "cwmr": cwmr_std.CWMR_STD, "eg": eg.EG, "sp": sp.SP, "ubah": ubah.UBAH,
         # "wmamr": wmamr.WMAMR}


def execute_multi_backtest(config, device):
    from pgportfolio.learn.nnagent import NNAgent
    ############################## Own config
    config["input"]["test_portion"] = 0.5
    config["input"]["silent"] = True
    for period in [300,600,900,1800,3600,7200,14400,2700,720,420,120,240,360,1200,1500,5400,9000,10800,540,480,660,180,60]:#[1200,360,7200,14400,600,480,2700,18000][420,540,1200,10800,300,900,1800,720]:#[180,420,540,1200,10800]:#[240,300,900,1800,720]:#[1800,300,60,180,120,900,3600,7200,14400,1200,600,2700]:
        config["input"]["global_period"] = period
        for (start, end) in [("2017/01/01", "2018/03/01"), ("2018/01/03", "2018/01/17"), ("2018/01/22", "2018/02/06"), ("2018/02/15", "2018/03/01"), ("2018/02/22", "2018/03/08"), ("2017/11/08", "2018/03/08"), ("2017/07/01", "2017/11/01")]:
            name = "./eval/ev" + str(period) + "-" + start.split('/')[0] + start.split('/')[1] + \
                                   start.split('/')[2] + "-" + end.split('/')[0] + \
                                   end.split('/')[1] + end.split('/')[2] + ".csv"
            list_algos = [6, 7, 8, 9, 10, 11, 12, 13, 14]
            list_coin_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            max_coins = max(list_coin_num)
            config["input"]["coin_number"] = max_coins
            config["input"]["start_date"] = start
            config["input"]["end_date"] = end
            if os.path.exists(name):
                results = pd.read_csv(name, index_col="Unnamed: 0")
            else:
                results = pd.DataFrame(index=list_algos, columns=list(map(str, list_coin_num)))
            print("\n" + "=" * 45)
            print("Backtesting on dates : {} - {}".format(start, end))
            print("Period :", period)
            print("-" * 45)
            if not results.isnull().any().any():
                print("Already tested")
                print("=" * 45)
            else:
                print("Loading {} coin{}...".format(max_coins, ("s" * min(1, max_coins))))
                print("=" * 45)
                backtester = BackTest(config, agent=None, agent_type="nn", net_dir=None)
                for coin_number in list_coin_num:
                    try:
                        if results.isnull()[str(coin_number)].any():
                            confirmed_coin_number = backtester.change_portfolio(coin_number)
                            print("\n" + "="*32)
                            print("Backtesting with", confirmed_coin_number, "coin{}...".format("s" * min(1,confirmed_coin_number-1)))
                            print("Dates : {} - {}".format(start,end))
                            print("Period :",period)
                            print("=" * 32)
                            for algo in list_algos:
                                if results.isnull()[str(coin_number)][algo]:
                                    print("=" * 30)
                                    print("Algorithm", algo)
                                    print("-" * 30)
                                    backtester.change_agent(algo,device)
                                    result = backtester.start_trading()
                                    results[str(coin_number)][algo] = result
                                    results.to_csv("./eval/eval.csv")
                                    results.to_csv(name)
                                    print(results)
                    except:
                        pass


def execute_backtest(algo, config):
    """
    @:param algo: string representing the name the name of algorithms
    @:return: numpy array of portfolio changes
    """
    print("Algorithm", algo)
    agent, agent_type, net_dir = _construct_agent(algo)
    backtester = BackTest(config, agent=agent, agent_type=agent_type, net_dir=net_dir)
    backtester.start_trading()
    return backtester.test_pc_vector

def execute_testtrade(algo, config):
    """
    @:param algo: string representing the name the name of algorithms
    @:return: numpy array of portfolio changes
    """
    agent, agent_type, net_dir = _construct_agent(algo)
    testtrader = TestTrade(config, agent=agent, agent_type=agent_type, net_dir=net_dir)
    testtrader.start_trading()
    return testtrader.test_pc_vector

def execute_realtrade(algo, config):
    """
    @:param algo: string representing the name the name of algorithms
    @:return: numpy array of portfolio changes
    """
    agent, agent_type, net_dir = _construct_agent(algo)
    realtrader = RealTrade(config, agent=agent, agent_type=agent_type, net_dir=net_dir)
    realtrader.start_trading()
    return realtrader.test_pc_vector

def _construct_agent(algo):
    if algo.isdigit():
        agent = None
        agent_type = "nn"
        net_dir = "./train_package/" + algo + "/netfile"
    elif algo in ALGOS:
        print("traditional agent")
        agent = ALGOS[algo]()
        agent_type = "traditional"
        net_dir = None
    else:
        agent = None
        agent_type = "nn"
        net_dir = "./train_package/" + algo + "/netfile"
        # message = "The algorithm name "+algo+" is not support. Supported algos " \
        #                                      "are " + str(list(ALGOS.keys()))
        # raise LookupError(message)
    return agent, agent_type, net_dir
