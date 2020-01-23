import numpy as np
import argparse
import json

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--data_type', type=str, default='0entropy',
                        help='the type of data that needs to be generated')
    parser.add_argument('--num_samples', type=int, default=10000000,
                        help='length of the sequence to be generated')
    parser.add_argument('--markovity', type=int, default=100,
                        help='Step for Markovity')
    parser.add_argument('--file_name', type=str, default='files_to_be_compressed/test.txt',
                        help='The name of the output file')
    parser.add_argument('--info_file', type=str, default='input_info.txt',
                        help='Name of the info file')
    parser.add_argument('--p1', type=float, default=0.5,
                        help='the probability for the entire sequence, or the base')
    parser.add_argument('--n1', type=float, default=0.1,
                        help='the probability for the entire sequence, or the base')
    
    return parser


# Computes the binary entropy
def entropy_iid(prob):
    p1 = prob
    p0 = 1.0 - prob
    H = -(p1*np.log(p1) + p0*np.log(p0))
    H /= np.log(2.0)
    return H

def main():
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    FLAGS.p0 = 1.0 - FLAGS.p1
    FLAGS.n0 = 1.0 - FLAGS.n1
    _keys = ["data_type","p1","n1"]

    data = np.empty([FLAGS.num_samples,1],dtype='S1')
    #print data.shape
    
    if FLAGS.data_type=='iid':
        #Generate data
        data = np.random.choice(['a', 'b'], size=(FLAGS.num_samples,1), p=[FLAGS.p0, FLAGS.p1])
        FLAGS.Entropy = entropy_iid(FLAGS.p1)
        _keys.append("Entropy")
 
    elif FLAGS.data_type=='0entropy':
        data[:FLAGS.markovity,:] = np.random.choice(['a', 'b'], size=(FLAGS.markovity,1), p=[FLAGS.p0, FLAGS.p1])
        for i in range(FLAGS.markovity, FLAGS.num_samples):
            if data[i-1] == data[i-FLAGS.markovity]:
                data[i] = 'a'
            else:
                data[i] = 'b'
        FLAGS.Entropy = 0
        _keys.append("Entropy")
        _keys.append("markovity")
    
    elif FLAGS.data_type=='HMM':
        data[:FLAGS.markovity,:] = np.random.choice(['a', 'b'], size=(FLAGS.markovity,1), p=[FLAGS.p0, FLAGS.p1])
        l1 = np.random.choice(['a','b'], size=(FLAGS.num_samples), p=[FLAGS.n0, FLAGS.n1])
        l2 = np.random.choice(['b','a'], size=(FLAGS.num_samples), p=[FLAGS.n0, FLAGS.n1])
        for i in range(FLAGS.markovity, FLAGS.num_samples):
            if data[i-1] == data[i-FLAGS.markovity]:
                data[i] = l1[i]
            else:
                data[i] = l2[i]
  
        FLAGS.Entropy = entropy_iid(FLAGS.n1) 
        _keys.append("Entropy")
        _keys.append("markovity")
        print "HMM Data generated ..." 

    np.savetxt(FLAGS.file_name,data,delimiter='', fmt='%c',newline='');
    
    #print _keys
    args = vars(FLAGS)
    info = { key : args[key] for key in _keys }
    #print info
    with open(FLAGS.info_file,"wb") as f:
        json.dump(info,f)

if __name__ == '__main__':
    main()
