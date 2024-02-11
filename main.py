from __future__ import absolute_import
import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime

from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.tools.trade import save_test_data
from pgportfolio.tools.shortcut import execute_backtest, execute_multi_backtest, execute_realtrade
from pgportfolio.resultprocess import plot


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="train, generate, download_data, backtest, multi_backtest",
                        metavar="MODE", default="trade")
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="repeat times of generating training subfolder",
                        default="6")
    parser.add_argument("--algo",
                        help="algo name or indexes of training_package ",
                        dest="algo",default="11")
    parser.add_argument("--algos",
                        help="algo names or indexes of training_package, seperated by \",\"",
                        dest="algos",default="8,9,99,10,11,12,13,14")
    parser.add_argument("--steps",
                        help="total steps to trade real time: >=1",
                        dest="steps")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="device to be used to train")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="folder(int) to load the config, neglect this option if loading from ./pgportfolio/net_config")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + "train_package"):
        os.makedirs("./" + "train_package")
    if not os.path.exists("./" + "database"):
        os.makedirs("./" + "database")

    if options.mode == "train":
        import pgportfolio.autotrain.training
        #if not options.algo or options.algo =='1':
        pgportfolio.autotrain.training.train_all(int(options.processes), options.device)
        # else:
        #     for folder in options.train_floder:
        #         raise NotImplementedError()
    elif options.mode == "generate":
        import pgportfolio.autotrain.generate as generate
        logging.basicConfig(level=logging.INFO)
        generate.add_packages(load_config(), int(options.repeat))
    elif options.mode == "download_data":
        from pgportfolio.marketdata.datamatrices import DataMatrices
        with open("./pgportfolio/net_config.json") as file:
            config = json.load(file)
        config = preprocess_config(config)
        start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
        end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
        DataMatrices(market=config["input"]["market"],
                     exchange=config["input"]["exchange"],
                     start=start,
                     end=end,
                     feature_number=config["input"]["feature_number"],
                     window_size=config["input"]["window_size"],
                     online=True,
                     period=config["input"]["global_period"],
                     granularity=config["input"]["trading_granularity"],
                     volume_average_days=config["input"]["volume_average_days"],
                     coin_filter=config["input"]["coin_number"],
                     is_permed=config["input"]["is_permed"],
                     test_portion=config["input"]["test_portion"],
                     portion_reversed=config["input"]["portion_reversed"])
    elif options.mode == "backtest":
        config = _config_by_algo(options.algo)
        config["input"]["simulation"] = True
        config["input"]["is_backtest"] = True
        _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
        execute_backtest(options.algo, config)
    elif options.mode == "multi_backtest":
        config = _config_by_algo(options.algo)
        config["input"]["silent"] = True
        config["input"]["simulation"] = True
        config["input"]["is_backtest"] = True
        _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
        execute_multi_backtest(config, options.device)
    elif options.mode == "trade":
        print("Algorithm", options.algo)
        config = _config_by_algo(options.algo)
        if config["input"]["simulation"] == True:
            _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "tradelog")
        else:
            _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "livelog")
        config["input"]["live_trading"]= True
        config["input"]["test_portion"]= 0.5
        if config["input"]["simulation"]:
            config["input"]["warmup"] = False
            config["input"]["silent"] = False
        execute_realtrade(options.algo, config)
    elif options.mode == "save_test_data":
        # This is used to export the test data
        save_test_data(load_config(options.folder))
    elif options.mode == "plot":
        logging.basicConfig(level=logging.INFO)
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.plot_backtest(load_config(), algos, labels)
    elif options.mode == "table":
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.table_backtest(load_config(), algos, labels, format=options.format)

def _set_logging_by_algo(console_level, file_level, algo, name):
    if algo.isdigit():
            logging.basicConfig(filename="./train_package/"+algo+"/"+name,
                                level=file_level)
            console = logging.StreamHandler()
            console.setLevel(console_level)
            logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=console_level)


def _config_by_algo(algo):
    """
    :param algo: a string represent index or algo name
    :return : a config dictionary
    """
    if not algo:
        raise ValueError("please input a specific algo")
    elif algo.isdigit():
        config = load_config(algo)
    else:
        config = load_config()
    return config

if __name__ == "__main__":
    main()
