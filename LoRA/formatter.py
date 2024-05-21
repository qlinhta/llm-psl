import json
import torch
import random
from torch.utils.data import Dataset

class FormatConverter:
    def __init__(self):
        pass
    
    def e2e_format_convert(self, read_file, write_file):
        with open(read_file, "r", encoding="utf8") as reader, \
             open(write_file, "w", encoding="utf8") as writer:
            for line in reader:
                items = line.strip().split("||")
                context = items[0]
                completion = items[1].strip("\n")
                x = {"context": context, "completion": completion}
                writer.write(json.dumps(x) + "\n")

    def dart_format_convert(self, read_file, write_file):
        with open(read_file, "r", encoding="utf8") as reader, \
             open(write_file, "w", encoding="utf8") as writer:
            lines_dict = json.load(reader)

            full_rela_lst = []
            full_src_lst = []
            full_tgt_lst = []
            unique_src = 0

            for example in lines_dict:
                rela_lst = []
                temp_triples = ''
                for i, tripleset in enumerate(example['tripleset']):
                    subj, rela, obj = tripleset
                    rela = rela.lower()
                    rela_lst.append(rela)
                    if i > 0:
                        temp_triples += ' | '
                    temp_triples += '{} : {} : {}'.format(subj, rela, obj)

                unique_src += 1

                for sent in example['annotations']:
                    full_tgt_lst.append(sent['text'])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)

            print('unique source is', unique_src)

            for src, tgt in zip(full_src_lst, full_tgt_lst):
                x = {"context": src, "completion": tgt}
                writer.write(json.dumps(x) + "\n")

    def webnlg_format_convert(self, read_file, write_file):
        with open(read_file, "r", encoding="utf8") as reader, \
             open(write_file, "w", encoding="utf8") as writer:
            lines_dict = json.load(reader)

            full_rela_lst = []
            full_src_lst = []
            full_tgt_lst = []
            full_cate_lst = []

            seen = [
                'Airport', 
                'Astronaut', 
                'Building', 
                'City', 
                'ComicsCharacter', 
                'Food', 
                'Monument', 
                'SportsTeam', 
                'University', 
                'WrittenWork'
            ]

            cate_dict = {}
            for i, example in enumerate(lines_dict['entries']):
                sents = example[str(i + 1)]['lexicalisations']
                triples = example[str(i + 1)]['modifiedtripleset']
                cate = example[str(i + 1)]['category']

                if cate not in cate_dict:
                    cate_dict[cate] = 0
                cate_dict[cate] += 1

                rela_lst = []
                temp_triples = ''
                for i, tripleset in enumerate(triples):
                    subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                    rela_lst.append(rela)
                    if i > 0:
                        temp_triples += ' | '
                    temp_triples += '{} : {} : {}'.format(subj, rela, obj)

                for sent in sents:
                    if sent["comment"] == 'good':
                        full_tgt_lst.append(sent['lex'])
                        full_src_lst.append(temp_triples)
                        full_rela_lst.append(rela_lst)
                        full_cate_lst.append(cate)

            for cate in cate_dict:
                print('cate', cate, cate_dict[cate])

            for src, tgt, cate in zip(full_src_lst, full_tgt_lst, full_cate_lst):
                x = {"context": src, "completion": tgt, "cate": cate in seen}
                writer.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    pass

    # converter = FormatConverter()
    # converter.e2e_format_convert("data/e2e/train.txt", "data/e2e/e2e_train_formatted.jsonl")
    # converter.e2e_format_convert("data/e2e/test.txt", "data/e2e/e2e_test_formatted.jsonl")

    # converter.dart_format_convert("data/dart/dart-v1.1.1-full-train.json", "data/dart/dart_train_formatted.jsonl")
    # converter.dart_format_convert("data/dart/dart-v1.1.1-full-test.json", "data/dart/dart_test_formatted.jsonl")

    # converter.webnlg_format_convert("data/webnlg_challenge_2017/train.json", "data/webnlg_challenge_2017/webnlg_train_formatted.jsonl")
    # converter.webnlg_format_convert("data/webnlg_challenge_2017/test.json", "data/webnlg_challenge_2017/webnlg_test_formatted.jsonl")

