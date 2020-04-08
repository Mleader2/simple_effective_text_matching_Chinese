# coding=utf-8
import os
import sys
import json5
from pprint import pprint
from src.utils import params, loader
from src.demoer import Demoer
from src.utils.metrics import registry as metrics
from curLine_file import curLine

def main():
    argv = sys.argv
    host_name = sys.argv[2]
    model_id = "benchmark-4"
    out_dir = "/home/%s/Mywork/corpus/Chinese_QA/" % host_name
    checkpoint_dir = "/home/cloudminds/PycharmProjects/simple_effective_text_matching_Chinese/models/%s" % model_id # % host_name
    print(curLine(), "argv:", argv, "host_name:", host_name)

    if len(argv) == 3:
        arg_groups = params.parse(sys.argv[1])

        args,config = arg_groups[0]
        print(curLine(), args, "config:", config)

        args.data_dir = os.path.join(out_dir, args.data_dir)

        args.pretrained_embeddings = os.path.join("/home/%s/Word2Vector/Chinese" % host_name,
                                                  args.pretrained_embeddings)
        demoer = Demoer(args, checkpoint_dir)
        predictions, probabilities, inference_time = demoer.serve(dev=[{'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'}])
        test(args, config, demoer) # 批量测试

        # corpus_list = []
        # predict_batchsize = 30
        # for i in range(predict_batchsize):
        #     corpus_list.append({'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'})
        #     batch = corpus_list
        #
        # inference_time_sum =0
        # cishu = 10
        # for i in range(cishu):
        #     predictions, probabilities, inference_time = demoer.serve(dev=batch)
        #     inference_time_sum += inference_time
        #     print(curLine(), inference_time)
        #     print(curLine(), "predictions:",predictions[0])
        #     print(curLine(), "probabilities:", probabilities[0])
        # print(curLine(), "inference_time=%f ms" % (inference_time_sum/cishu ))

    elif len(argv) == 4 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print(curLine(), 'Usage: "python train.py configs/xxx.json5 host_name"')

def test(args, config, demoer):

    data_dir = args.data_dir
    dev = loader.load_data(data_dir, args.eval_file)
    print(curLine(), len(dev), "dev:", dev[0]) # dev: list of dict, such as {'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'}
    targets = []
    for sample in dev:
        targets.append(int(sample['target']))
    print(curLine(), "targets:", targets[:10])
    predictions, probabilities, inference_time = demoer.serve(dev=dev, batch_size=128*3)
    outputs = {
        'target': targets,
        'prob': probabilities,
        'pred': predictions,
        'args': args,
    }
    print(curLine(), "predictions:", predictions[:10])
    print(curLine(), "probabilities:", probabilities[:10])
    # total_loss = sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses)
    states = {
        'inference_time': inference_time/len(targets)
    }
    for metric in args.watch_metrics:
        if metric not in states:  # multiple metrics could be computed by the same function
            states.update(metrics[metric](outputs))
    print(curLine(), "stats:", states)
    with open('models/log.jsonl', 'a') as f:
        f.write(json5.dumps({
            'data': os.path.basename(args.data_dir),
            'params': config,
            'state': states}))
        f.write('\n')

if __name__ == '__main__':
    main()
