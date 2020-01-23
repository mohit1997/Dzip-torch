wget http://sun.aei.polsl.pl/~sdeor/corpus/mozilla.bz2
bzip2 -d mozilla.bz2
mkdir -p files_to_be_compressed
mv mozilla files_to_be_compressed/
