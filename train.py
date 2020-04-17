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
    host_name = sys.argv[2]
    if len(argv) == 3:
        arg_groups = params.parse(sys.argv[1], host_name, mode="train")
        test_score_sum = 0.0
        max_test_score = 0.0
        experiment_times = 0
        eval_score_list = []
        best_experiment_times = None
        for args, config in arg_groups:
            if not os.path.exists(args.summary_dir):
                os.makedirs(args.summary_dir)
            args.pretrained_embeddings= os.path.join("/home/%s/Word2Vector/Chinese"% host_name, args.pretrained_embeddings)
            # print(curLine(), "args.data_dir:%s, args.output_dir:%s" % (args.data_dir, args.output_dir))
            trainer = Trainer(args)
            states,best_eval_score = trainer.train(experiment_times)
            eval_score_list.append(best_eval_score)
            test_score_sum += best_eval_score
            if max_test_score < best_eval_score:
                max_test_score = best_eval_score
                best_experiment_times = experiment_times
            experiment_times += 1
            print(curLine(), "experiment_times=%d/%d, best_experiment_times=%d, ave_test_score=%f, max_test_score=%f"
                  % (experiment_times, len(arg_groups), best_experiment_times, test_score_sum / experiment_times, max_test_score))
            with open('%s/log.jsonl'%args.output_dir, 'a') as f:
                f.write(json5.dumps({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
            print(curLine(), "eval_score_list:", eval_score_list, eval_score_list.index(max_test_score))
    else:
        print(curLine(), 'Usage: "python train.py configs/xxx.json5 host_name"')


if __name__ == '__main__':
    main()
