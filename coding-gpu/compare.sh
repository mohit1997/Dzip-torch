#!/bin/bash
# Based on https://unix.stackexchange.com/questions/397655/two-files-comparison-in-bash-script

file1=$1
file2=$2

if cmp -s "$file1" "$file2"; then
    printf 'The file "%s" is the same as "%s"\n' "$file1" "$file2"
else
    printf 'The file "%s" is different from "%s"\n' "$file1" "$file2"
fi
