# coding=utf-8
import os
import sys
import json5
from pprint import pprint
from src.utils import params, loader
from src.demoer import Demoer
from curLine_file import curLine

def main():
    argv = sys.argv
    host_name = sys.argv[2]
    print(curLine(), "argv:", argv, "host_name:", host_name)

    if len(argv) == 3:
        arg_groups = params.parse(sys.argv[1])

        args,config = arg_groups[0]
        print(curLine(), args, "config:", config)
        out_dir = "/home/%s/Mywork/corpus/Chinese_QA/" % host_name
        args.data_dir = os.path.join(out_dir, args.data_dir)
        args.pretrained_embeddings = os.path.join("/home/%s/Word2Vector/Chinese" % host_name,
                                                  args.pretrained_embeddings)
        demoer = Demoer(args)
        predictions, probabilities, inference_time = demoer.serve(dev=[{'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'}])
        # test(args, config, demoer) # 批量测试

        corpus_list = []
        predict_batchsize = 30
        for i in range(predict_batchsize):
            corpus_list.append({'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'})
            batch = corpus_list

        inference_time_sum =0
        cishu = 10
        for i in range(cishu):
            predictions, probabilities, inference_time = demoer.serve(dev=batch)
            inference_time_sum += inference_time
            print(curLine(), inference_time)
            print(curLine(), "predictions:",predictions[0])
            print(curLine(), "probabilities:", probabilities[0])
        print(curLine(), "inference_time=%f ms" % (inference_time_sum/cishu ))

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

    predictions, probabilities, inference_time = demoer.serve(corpus_list=dev)
    print(curLine(), inference_time, type(predictions))
    states = {}
    with open('models/log.jsonl', 'a') as f:
        f.write(json5.dumps({
            'data': os.path.basename(args.data_dir),
            'params': config,
            'state': states}))
        f.write('\n')

if __name__ == '__main__':
    main()
