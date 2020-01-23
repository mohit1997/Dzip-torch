wget http://sun.aei.polsl.pl/~sdeor/corpus/webster.bz2
bzip2 -d webster.bz2
mkdir -p files_to_be_compressed
mv webster files_to_be_compressed/
