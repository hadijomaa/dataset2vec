from d2v import D2V
from runner import Runner

def main(config,dataset,seed):
    if dataset=='toy':
        from toy import Toy
        ds  = Toy(seed=seed,split=config['split'])
    elif dataset=='uci':
        from uci import UCI
        ds  = UCI(seed=seed,split=config['split'])
    lfp = D2V(config=config)
    optimizer = Runner(config=config,dataset=ds,model=lfp)
    optimizer.run(dataset=dataset)
#    phi = optimizer.summarize()
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--dataset', help='Which Configuration', type=str,default='toy')
    parser.add_argument('--seed', help='Which Configuration', type=int,default=0)
    args = parser.parse_args()
    from config import config
    main(config=config,dataset=args.dataset,seed=args.seed)
