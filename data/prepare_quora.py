# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from tqdm import tqdm

raw_folder = "/home/cloudminds/Mywork/corpus/English/glue_corpus"
print('processing quora, folder:%s' % raw_folder)
os.makedirs('quora', exist_ok=True)
# use the partition on https://zhiguowang.github.io
for split in ('train', 'dev', 'test'):
    raw_file_name = '%s/Quora_question_pair_partition/%s.tsv' % (raw_folder,split)
    save_file_name = '%s/quora/%s.txt' % (raw_folder, split)
    print("raw_file_name:", raw_file_name, "save_file_name:", save_file_name)
    with open(raw_file_name) as f, \
            open(save_file_name, 'w') as fout:
        n_lines = 0
        for _ in f:
            n_lines += 1
        f.seek(0)
        for line in tqdm(f, total=n_lines, leave=False):
            elements = line.rstrip().split('\t')
            fout.write('{}\t{}\t{}\n'.format(elements[1], elements[2], int(elements[0])))
