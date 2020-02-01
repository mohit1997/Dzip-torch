import sys
import numpy as np
import json
import argparse
import os

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='files_to_be_compressed/hmm40.txt',
                        help='The name of the input file')
    
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()

input_file = FLAGS.file_name
base_name = os.path.basename(input_file)
param_file = "params_" + os.path.splitext(base_name)[0]
output_file = os.path.splitext(base_name)[0]

with open(input_file, 'r', encoding='latin-1') as fp:
    data = fp.read()

print("Seq Length {}".format(len(data)))
vals = list(set(data))
vals.sort()
print(vals)

char2id_dict = {c: i for (i,c) in enumerate(vals)}
id2char_dict = {i: c for (i,c) in enumerate(vals)}

params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}
with open(param_file, 'w') as f:
    json.dump(params, f, indent=4)

print(char2id_dict)
print(id2char_dict)

out = [char2id_dict[c] for c in data]
integer_encoded = np.array(out)

np.save(output_file, integer_encoded)
