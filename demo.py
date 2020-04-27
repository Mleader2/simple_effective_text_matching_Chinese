# coding=utf-8
# can be used for demo or test
import os
import sys
import json5
import codecs
from src.utils import params, loader
from src.demoer import Demoer
from src.utils.metrics import registry as metrics
from curLine_file import curLine, normal_transformer

def main():
    argv = sys.argv
    host_name = argv[2]
    model_id = argv[3]
    print(curLine(), "argv:", argv)
    arg_groups = params.parse(argv[1], host_name, mode="test")
    args, config = arg_groups[0]
    args.output_dir = "/home/%s/Mywork/model/qa_model_dir/on_test/block1-layer1-hidden100-acc=85.31" % (host_name)  # TODO
    args.output_dir = "/home/%s/Mywork/model/qa_model_dir/part_chatcorpus_model/block1-layer1-hidden100-normal-acc80.57" % host_name

    args.data_dir = os.path.join("/home/%s/Mywork/corpus/Chinese_QA" % host_name, args.data_dir)
    checkpoint_dir = os.path.join(args.output_dir, model_id)

    if len(argv) == 5:
        args.eval_file = argv[4]
    demoer = Demoer(args, checkpoint_dir)
    sample = {'text1':"请问谁有狂三这张高清的电影资源？", 'text2': '这张高清图，谁有狂三这张高清的请问谁有狂三这张高清的电影资源？'}
    predictions, probabilities, inference_time = demoer.serve(dev=[sample])

    test(args, config, demoer) # 批量测试

    # predict_batchsize = 30
    # infer_flag = True # False  #
    #
    # if infer_flag:
    #     text2_list = []
    #     for i in range(predict_batchsize):
    #         text2_list.append(sample['text2'])
    #     batch = [{'text1': sample['text1'], 'text2_list': text2_list}]
    # else:
    #     batch = []
    #     for i in range(predict_batchsize):
    #         batch.append({'text1': sample['text1'], 'text2': sample['text2']})
    #
    # inference_time_sum1 = 0
    # for i in range(20):
    #     predictions, probabilities, inference_time = demoer.serve(dev=batch, infer_flag=infer_flag)
    #     inference_time_sum1 += inference_time
    # inference_time_sum =0
    # cishu = 400
    # for i in range(cishu):
    #     predictions, probabilities, inference_time = demoer.serve(dev=batch, infer_flag=infer_flag)
    #     inference_time_sum += inference_time
    #     # print(curLine(), inference_time)
    #     # print(curLine(), "predictions:",predictions)
    #     # print(curLine(), "probabilities:", probabilities[0])
    # print(curLine(), "inference_time1=%f ms, inference_time=%f ms" % (inference_time_sum1/20, inference_time_sum/cishu ))
    # print(curLine(), f"infer_flag:{infer_flag}, predict_batchsize={predict_batchsize}, inference_time={inference_time_sum/cishu}ms")

def test(args, config, demoer):
    dev = loader.load_data(args.data_dir, args.eval_file)
    targets = []
    for sample in dev:
        targets.append(int(sample['target']))
    predictions, probabilities, inference_time = demoer.serve(dev=dev, batch_size=384)

    if "train" in args.eval_file:  # 将模型的置信度保存到文件
        with open(os.path.join(args.data_dir, "%s.txt" % args.eval_file), "r") as fr:
            lines = fr.readlines()
        assert len(lines) == len(probabilities), 'number of lines is %d, number of probabilities is %d' % (len(lines), len(probabilities))
        save_file = os.path.join(args.data_dir, "%s_score.txt" % args.eval_file)
        with open(save_file, "w") as writer:
            for line, prediction, prob in zip(lines, predictions, probabilities):
                writer.write("%s\t%f\n" % (line.strip(), prob[1]))
        print(curLine(), "save %d results to %s" % (len(probabilities), save_file))

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
