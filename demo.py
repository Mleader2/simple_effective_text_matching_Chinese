# coding=utf-8
# can be used for demo or test
import os
import sys
import json5
from src.utils import params, loader
from src.demoer import Demoer
from src.utils.metrics import registry as metrics
from curLine_file import curLine

def main():
    argv = sys.argv
    host_name = sys.argv[2]
    model_id = sys.argv[3]
    print(curLine(), "argv:", argv)
    arg_groups = params.parse(sys.argv[1], host_name, mode="test")
    args, config = arg_groups[0]
    out_dir = "/home/%s/Mywork/corpus/Chinese_QA/" % host_name
    # args.output_dir = "/home/%s/Mywork/model/qa_model_dir/chat_corpus_all" % (host_name) # --good TODO
    args.output_dir = "/home/%s/Mywork/model/qa_model_dir/on_dev/block1-layer1-hidden150" % host_name

    args.data_dir = os.path.join(out_dir, args.data_dir)
    checkpoint_dir = os.path.join(args.output_dir, model_id)
    demoer = Demoer(args, checkpoint_dir)
    sample = {'text1':"请问谁有狂三这张高清的电影资源？", 'text2': '这张高清图，谁有狂三这张高清的请问谁有狂三这张高清的电影资源？'}
    predictions, probabilities, inference_time = demoer.serve(dev=[sample])
    # test(args, config, demoer) # 批量测试

    predict_batchsize = 30
    infer_flag = True # False  #
    print(curLine(), "infer_flag:", infer_flag, ",predict_batchsize=", predict_batchsize)
    if infer_flag:
        text2_list = []
        for i in range(predict_batchsize):
            text2_list.append(sample['text2'])
        batch = [{'text1': sample['text1'], 'text2_list': text2_list}]
    else:
        batch = []
        for i in range(predict_batchsize):
            batch.append({'text1': sample['text1'], 'text2': sample['text2']})

    inference_time_sum1 = 0
    for i in range(20):
        predictions, probabilities, inference_time = demoer.serve(dev=batch, infer_flag=infer_flag)
        inference_time_sum1 += inference_time
    inference_time_sum =0
    cishu = 400
    for i in range(cishu):
        predictions, probabilities, inference_time = demoer.serve(dev=batch, infer_flag=infer_flag)
        inference_time_sum += inference_time
        # print(curLine(), inference_time)
        # print(curLine(), "predictions:",predictions)
        # print(curLine(), "probabilities:", probabilities[0])
    print(curLine(), "inference_time1=%f ms, inference_time=%f ms" % (inference_time_sum1/20, inference_time_sum/cishu ))


def test(args, config, demoer):
    data_dir = args.data_dir
    dev = loader.load_data(data_dir, args.eval_file)#[-20:]
    # print(curLine(), len(dev), "dev:", dev) # dev: list of dict, such as {'text1':'谁有狂三这张高清的', 'text2': '这张高清图，谁有'}
    targets = []
    for sample in dev:
        targets.append(int(sample['target']))
    predictions, probabilities, inference_time = demoer.serve(dev=dev, batch_size=128*3)
    outputs = {
        'target': targets,
        'prob': probabilities,
        'pred': predictions,
        'args': args,
    }
    # total_loss = sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses)
    states = {
        'inference_time': inference_time/len(targets)
    }
    for metric in args.watch_metrics:
        if metric not in states:  # multiple metrics could be computed by the same function
            states.update(metrics[metric](outputs))
    print(curLine(), "stats:", states)
    with open('%s/log.jsonl'%args.output_dir, 'a') as f:
        f.write(json5.dumps({
            'data': os.path.basename(args.data_dir),
            'params': config,
            'state': states}))
        f.write('\n')

if __name__ == '__main__':
    main()
