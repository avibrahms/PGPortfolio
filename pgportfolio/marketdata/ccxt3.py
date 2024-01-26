import ccxt


def get_pairs(e):
    e.load_markets()
    pairs=list(e.markets.keys())
    return pairs

def give_common_pairs(e1,e2):
    pairs_e1 = get_pairs(e1)
    pairs_e2 = get_pairs(e2)
    common = []
    for p1 in pairs_e1:
        for p2 in pairs_e2:
            if p1==p2:
                common.append(p1)
    return common



common = give_common_pairs(ccxt.binance(),ccxt.poloniex())

print(len(common),common)