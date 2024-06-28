#!/bin/bash
mkdir -p data/har
pushd data/har
wget --no-check-certificate https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip
unzip smartphone+based+recognition+of+human+activities+and+postural+transitions.zip
rm -f smartphone+based+recognition*
popd
