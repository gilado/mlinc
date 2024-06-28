#!/bin/bash
mkdir -p data/jvowels
pushd data/jvowels
wget --no-check-certificate https://archive.ics.uci.edu/static/public/128/japanese+vowels.zip
unzip japanese+vowels.zip 
rm -f japanese+vowels.zip
chmod 444 *
popd
