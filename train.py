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
    host_name = sys.argv[2]
    print(curLine(), "argv:", argv, "host_name:", host_name)
    out_dir = "/home/%s/Mywork/corpus/Chinese_QA/" % host_name

    if len(argv) == 3:
        arg_groups = params.parse(sys.argv[1])
        test_score_sum = 0.0
        max_test_score = 0.0
        experiment_times = 0
        eval_score_list = []
        best_experiment_times = None
        for args, config in arg_groups:
            args.summary_dir = args.summary_dir.replace("models", "/home/%s/Mywork/model/qa_model_dir" %host_name)
            print(curLine(), "args.summary_dir:", args.summary_dir)
            if not os.path.exists(args.summary_dir):
                os.makedirs(args.summary_dir)
            args.data_dir = os.path.join(out_dir, args.data_dir)
            args.pretrained_embeddings= os.path.join("/home/%s/Word2Vector/Chinese"% host_name, args.pretrained_embeddings)
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
            with open('models/log.jsonl', 'a') as f:
                f.write(json5.dumps({
                    'data': os.path.basename(args.data_dir),
                    'params': config,
                    'state': states,
                }))
                f.write('\n')
            print(curLine(), "eval_score_list:", eval_score_list, eval_score_list.index(max_test_score))

    elif len(argv) == 4 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print(curLine(), 'Usage: "python train.py configs/xxx.json5 host_name"')


if __name__ == '__main__':
    main()
