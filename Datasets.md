## Links to the Datasets and Trained Boostrap Models
| File | Link |Bootstrap Model|
|------|------|------|
|webster|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|[webster](./Models/webster.bootstrap)|
|mozilla|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|[mozilla](./Models/mozilla.bootstrap)|
|h. chr20|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz|[chr20](./Models/chr20_bstrap)|
|h. chr1|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz|[chr1](./Models/chr1_bstrap)|
|c.e. genome|ftp://ftp.ensembl.org/pub/release-97/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz|[celegchr](./Models/celegchr_bstrap)|
|ill-quality|http://bix.ucsd.edu/projects/singlecell/nbt_data.html|[phixq](./Models/phixq_truncated_bstrap)|
|text8|http://www.mattmahoney.net/dc/textdata.html|[text8](./Models/text8_bstrap)|
|enwiki9|http://www.mattmahoney.net/dc/textdata.html|[enwiki9](./Models/enwiki9.bootstrap)|
|np-bases|https://github.com/nanopore-wgs-consortium/NA12878|[npbases](./Models/npbases_bstrap)|
|np-quality|https://github.com/nanopore-wgs-consortium/NA12878|[npquals](./Models/npquals_bstrap)|
|num-control|https://userweb.cs.txstate.edu/~burtscher/research/datasets/FPdouble/|[model](./Models/num_control.trace.bootstrap)|
|obs-spitzer|https://userweb.cs.txstate.edu/~burtscher/research/datasets/FPdouble/|[model](./Models/obs_spitzer.trace.bootstrap)|
|msg-bt|https://userweb.cs.txstate.edu/~burtscher/research/datasets/FPdouble/|[model](./Models/msg_bt.trace.bootstrap)|
|audio|https://github.com/karolpiczak/ESC-50|[model](./Models/audio.bootstrap)|


## Dataset Description

| File | Length | Vocabulary Size | Brief Description |
|------|------|------|------| 	
|<em>webster</em>| 41.1M|  98 |  HTML data of the 1913 Webster Dictionary, from the Silesia corpus | 
|<em>text8<em>|100M|  27|   First 100M of English text (only) extracted from enwiki9| 
|<em>enwiki9<em>|500M|  206|  First 500M of the English Wikipedia dump on 2006| 		
|<em>mozilla<em>| 51.2M|  256|  Tarred executables of Mozilla 1.0, from the Silesia corpus| 
|<em>h. chr20<em>|  64.4M|  5|   Chromosome 20 of H. sapiens GRCh38 reference sequence | 
|<em>h. chr1<em>|  100M|  5|  First 100M bases of chromosome 1 of H. Sapiens GRCh38 sequence |
|<em>c.e. genome <em>|  100M|  4|  C. elegans whole genome sequence| 
|<em>ill-quality<em>| 100M|  4|  100MB of quality scores for PhiX virus reads sequenced with Illumina | 
|<em>np-bases<em>|300M|  5|  Nanopore sequenced reads (only bases) of a human sample (first 300M symbols) | 
|<em>np-quality<em>| 300M|  91|  Quality scores for nanopore sequenced human sample (first 300M symbols)| 
|<em>num-control<em>|159.5M|  256|  Control vector output between two minimization steps in weather-satellite data assimilation| 
|<em>obs-spitzer<em>|198.2M|  256|  Data from the Spitzer Space Telescope showing a slight darkening| 
|<em>msg-bt<em>|266.4M|  256|  NPB computational fluid dynamics pseudo-application bt| 	
|<em>audio<em>|264.6M|  256|  First 600 files (combined) in ESC Dataset for environmental sound classification|
|<em>XOR-k <em>|  10M|  2|  Pseudorandom sequence <img src="https://render.githubusercontent.com/render/math?math=S_%7Bn%2B1%7D%20%3D%20S_n%20%5Cbigoplus%20S_%7Bn-k%7D"> Entropy rate 0 bpc.| 
|<em>HMM-k<em>|  10M|  2|  Hidden Markov sequence <img src="https://render.githubusercontent.com/render/math?math=S_n%20%3D%20X_n%20%5Cbigoplus%20Z_n">, with <img src="https://render.githubusercontent.com/render/math?math=Z_n%20%5Csim%20Bern(0.1)">, <img src="https://render.githubusercontent.com/render/math?math=X_%7Bn%2B1%7D%20%3D%20X_n%20%5Cbigoplus%20X_%7Bn-k%7D"> Entropy rate 0.46899 bpc. | 

