#!/bin/bash
mkdir -p data/iris
pushd data/iris
wget --no-check-certificate  https://archive.ics.uci.edu/static/public/53/iris.zip
unzip iris.zip 
mv -f bezdekIris.data iris.csv
mv -f iris.names iris.txt
chmod 444 iris.csv iris.txt 
rm -f iris.data iris.zip Index 
popd
