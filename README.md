# Dzip
## improved general-purpose lossless compression based on novel neural network modeling
#### Arxiv: https://arxiv.org/abs/1911.03572
## Description
DZip is a general lossless compressor for sequential data which uses NN-based modelling combined with arithmetic coding. We refer to the NN-based model as the "combined model", as it is composed of a bootstrap model and a supporter model. The bootstrap model is trained prior to compression on the data to be compressed, and the resulting model parameters (weights) are stored as part of the compressed output (after being losslessly compressed with BSC). The combined model is adaptively trained (bootstrap model parameters are fixed) while compressing the data, and hence its parameters do not need to be stored as part of the compressed output.

## Requirements
0. GPU (Cuda 9.0+)
1. Python3 (<= 3.6.8)
2. Numpy
3. Sklearn
4. Pytorch (gpu/cpu) 1.4


### Download and install dependencies
Download:
```bash
git clone https://github.com/mohit1997/Dzip-torch.git
```
To set up virtual environment and dependencies (on Linux):
```bash
cd DZip
python3 -m venv torch
source torch/bin/activate
bash install.sh
```

##### ENCODING-DECODING (uses cpu and slower)
<!-- 1. Go to [encode-decode](./encode-decode)
2. Place the parsed files in the directory files_to_be_compressed.
3. Run the following command -->

```bash 
cd coding-gpu
# Compress using the combined model (default usage of DZip)
bash compress.sh FILE.txt FILE.dzip com
# Compress using only the bootstrap model
bash compress.sh FILE.txt FILE.dzip bs
# Decompress using combined model (Only if compressed using combined mode)
bash decompress.sh FILE.dzip decom_FILE com MODEL_PATH
# Decompress using bootstrap model (Only if compressed using bs mode)
bash decompress.sh FILE.dzip decom_FILE bs MODEL_PATH
# Verify successful decompression
bash compare.sh FILE.txt decom_FILE
```

## Links to the Datasets and Trained Boostrap Models
| File | Link |Bootstrap Model|
|------|------|------|
|webster|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|[webster](./Models/webster_bstrap)|
|mozilla|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|[mozilla](./Models/mozilla_bstrap)|
|h. chr20|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz|[chr20](./Models/chr20_bstrap)|
|h. chr1|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz|[chr1](./Models/chr1_bstrap)|
|c.e. genome|ftp://ftp.ensembl.org/pub/release-97/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz|[celegchr](./Models/celegchr_bstrap)|
|ill-quality|http://bix.ucsd.edu/projects/singlecell/nbt_data.html|[phixq](./Models/phixq_truncated_bstrap)|
|text8|http://www.mattmahoney.net/dc/textdata.html|[text8](./Models/text8_bstrap)|
|enwiki9|http://www.mattmahoney.net/dc/textdata.html|[enwiki9](./Models/enwik9_bstrap)|
|np-bases|https://github.com/nanopore-wgs-consortium/NA12878|[npbases](./Models/npbases_bstrap)|
|np-quality|https://github.com/nanopore-wgs-consortium/NA12878|[npquals](./Models/npquals_bstrap)|

##
1. Go to [Datasets](./Datasets)
2. For real datasets, run
```bash
bash get_data.sh
```
3. For synthetic datasets, run
```bash
# For generating XOR-10 dataset
python generate_data.py --data_type 0entropy --markovity 10 --file_name files_to_be_compressed/xor10.txt
# For generating HMM-10 dataset
python generate_data.py --data_type HMM --markovity 10 --file_name files_to_be_compressed/hmm10.txt
```
4. This will generate a folder named `files_to_be_compressed`. This folder contains the parsed files which can be used to recreate the results in our paper.
