import transformers
from torch import tensor

tokenizer = transformers.T5Tokenizer.from_pretrained('allenai/unifiedqa-v2-t5-small-1251000')

import json

lens = set()

for split in ['trn','dev','tst']:
    for negation_type in ['original','logical_neg','semi_logical_neg','commonsense_contradict']:
        for line in open(f'{negation_type}/{negation_type}_{split}_formatted.json', 'r').readlines():
            lens.append(
                len(
                    tokenizer(
                        [
                            json.loads(line)['input']
                        ]
                    )['input_ids'][0]
                )
            )

print(f'FINAL LENS {sorted(lens)}')
'''
The longest input sequence is 69 tokens, so I will use a sequence length of 72
The longest answer sequence is 1 token, so I will use a sequence length of 2
'''