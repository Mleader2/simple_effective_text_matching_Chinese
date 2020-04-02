# coding=utf-8


import sys
from src.evaluator import Evaluator


def main():
    argv = sys.argv
    if len(argv) == 3:
        model_path, data_file = argv[1:]
        evaluator = Evaluator(model_path, data_file)
        evaluator.evaluate()
    else:
        print('Usage: "python evaluate.py $model_path $data_file"')


if __name__ == '__main__':
    main()
