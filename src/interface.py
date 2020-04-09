# coding=utf-8
import os
import random
import msgpack
from .utils.vocab import Vocab, Indexer
from .utils.loader import load_data, load_embeddings_English, load_embeddings_Chinese
from curLine_file import curLine

class Interface:
    def __init__(self, args, log=None):
        self.args = args
        # build/load vocab and target map
        vocab_file = os.path.join(args.output_dir, 'vocab.txt')
        target_map_file = os.path.join(args.output_dir, 'target_map.txt')
        if not os.path.exists(vocab_file):
            data = load_data(self.args.data_dir)
            self.target_map = Indexer.build((sample['target'] for sample in data), log=log)
            self.target_map.save(target_map_file)

            if self.args.language.lower() == "chinese":
                words = [word for sample in data   # Chinese
                                          for text in (sample['text1'], sample['text2'])
                                          for word in list(text)[:self.args.max_len]]
            else:
                words = [word for sample in data  # English
                                          for text in (sample['text1'], sample['text2'])
                                          for word in text.split()[:self.args.max_len]]
            # print(curLine(), type(words), "words:", len(words), words[-1])
            self.vocab = Vocab.build(words,
                                     lower=args.lower_case, min_df=self.args.min_df, log=log,
                                     pretrained_embeddings=args.pretrained_embeddings,
                                     dump_filtered=os.path.join(args.output_dir, 'filtered_words.txt'))
            self.vocab.save(vocab_file)
        else:
            self.target_map = Indexer.load(target_map_file)
            self.vocab = Vocab.load(vocab_file)
        args.num_classes = len(self.target_map)
        args.num_vocab = len(self.vocab)
        args.padding = Vocab.pad()

    def load_embeddings(self):
        """generate embeddings suited for the current vocab or load previously cached ones."""
        embedding_file = os.path.join(self.args.output_dir, 'embedding.msgpack')
        if not os.path.exists(embedding_file):
            if self.args.language == "chinese":
                embeddings = load_embeddings_Chinese(self.args.pretrained_embeddings, self.vocab,
                                         self.args.embedding_dim, mode=self.args.embedding_mode,
                                         lower=self.args.lower_case)
            else:
                embeddings = load_embeddings_English(self.args.pretrained_embeddings, self.vocab,
                                             self.args.embedding_dim, mode=self.args.embedding_mode,
                                             lower=self.args.lower_case)
            with open(embedding_file, 'wb') as f:
                msgpack.dump(embeddings, f)
        else:
            with open(embedding_file, 'rb') as f:
                embeddings = msgpack.load(f)
        return embeddings

    def pre_process(self, data, training=True, batch_size=None, infer_flag=False):
        if infer_flag:
            batch_result = []
            for sample in data:
                processed_text1, processed_len1 = self.process_sample(sample["text1"])
                processed_text2_list = []
                processed_len2_list = []
                for text2 in sample["text2_list"]:
                    processed_text2, processed_len2 = self.process_sample(text2)
                    processed_text2_list.append(processed_text2)
                    processed_len2_list.append(processed_len2)
                process_sample_dict = {"text1": [processed_text1], "len1": [processed_len1],
                                       "text2":processed_text2_list, "len2": processed_len2_list}
                min_len = max(processed_len1, max(processed_len2_list))
                batch = {key: self.padding(value, min_len=min_len) if key.startswith('text') else value
                         for key, value in process_sample_dict.items()}
                batch_result.append(batch)
        else:
            result = []
            for sample in data:
                processed_text1, processed_len1 = self.process_sample(sample["text1"])

                processed_text2, processed_len2 = self.process_sample(sample["text2"])
                process_sample_dict = {"text1": processed_text1, "len1": processed_len1,
                                       "text2":processed_text2, "len2": processed_len2}
                if 'target' in sample:
                    target = sample['target']
                    assert target in self.target_map
                    process_sample_dict['target'] = self.target_map.index(target)
                result.append(process_sample_dict)
            if training:
                result = list(filter(lambda x: x['len1'] < self.args.max_len and x['len2'] < self.args.max_len, result))
                if not self.args.sort_by_len:
                    return result
                result = sorted(result, key=lambda x: (x['len1'], x['len2'], x['text1']))
            if batch_size is None:
                batch_size = self.args.batch_size
            batch_result = [self.make_batch(result[i:i + batch_size]) for i in range(0, len(data), batch_size)]
        return batch_result

    def process_sample(self, text):
        if self.args.lower_case:
            text = text.lower()
        if self.args.language.lower() == "chinese":
            processed_text = [self.vocab.index(w) for w in list(text)[:self.args.max_len]]
        else:
            processed_text = [self.vocab.index(w) for w in text.split()[:self.args.max_len]],
        processed_len = len(processed_text)

        return processed_text, processed_len

    def shuffle_batch(self, data):
        data = random.sample(data, len(data))
        if self.args.sort_by_len:
            return data
        batch_size = self.args.batch_size
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return list(map(self.make_batch, batches))

    def make_batch(self, batch, with_target=True):
        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        if 'target' in batch and not with_target:
            del batch['target']

        # TODO TODO
        min_len = max(max(map(len, batch["text1"])), self.args.min_len)
        min_len = max(max(map(len, batch["text2"])), min_len)
        # print(curLine(), "min_len =", min_len) # TODO TODO
        batch = {key: self.padding(value, min_len=min_len) if key.startswith('text') else value
                 for key, value in batch.items()}
        return batch

    @staticmethod
    def padding(samples, min_len=1):
        max_len = max(max(map(len, samples)), min_len)
        batch = [sample + [Vocab.pad()] * (max_len - len(sample)) for sample in samples]
        return batch

    def post_process(self, output):
        final_prediction = []
        for prob in output:
            idx = max(range(len(prob)), key=prob.__getitem__)
            target = self.target_map[idx]
            final_prediction.append(target)
        return final_prediction
