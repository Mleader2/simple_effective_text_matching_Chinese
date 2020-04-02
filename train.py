# coding=utf-8


import os
import sys
import json5
from pprint import pprint
from src.utils import params
from src.trainer import Trainer
from curLine_file import curLine

def main():
    argv = sys.argv
    print(curLine(), "argv:", argv)
    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            trainer = Trainer(args)
            states = trainer.train()
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dumps({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print(curLine(), 'Usage: "python train.py configs/xxx.json5"')


if __name__ == '__main__':
    main()
