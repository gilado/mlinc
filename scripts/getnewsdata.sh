#!/bin/bash
mkdir -p data/news
if ! [ -d data/news/data ] ; then
  echo "The news dataset is available for download at "
  echo "https://www.kaggle.com/datasets/sbhatti/news-articles-corpus"
  echo "Place the compressed file in '$(pwd)/data/news' as"
  echo "news-articles-corpus.zip"
  echo "Alternatively, you can download it using the kaggle cli program."
  read -p "Press any key to continue, ^C to abort "
  pushd data/news
  # If the below command does not work, install kaggle cli by running
  # 'pip install kaggle'. 
  # If it is installed but is not found by this script, run "
  # 'pip uninstall kaggle' to find its location, and add it to the path
  if ! [ -f news-articles-corpus.zip ] ; then 
      echo "Downloading..."
      kaggle datasets download -d sbhatti/news-articles-corpus
  fi
  if [ -f news-articles-corpus.zip ] ; then 
      echo "Decompressing downloaded archive..."
      unzip news-articles-corpus.zip 
  else
      echo -e "\nnews-articles-corpus.zip does not exist.\n"
  fi
  popd
fi
if [ -d data/news/data ] ; then
  pushd data/news/data
  filetot=870521 # expected file count
  filecnt=$(ls -1 | grep article | wc -l)
  if [ "$filecnt" -eq "$filetot" ] ; then
      echo "Dataset downloaded"
  else
      echo "Dataset incomplete: expected $filetot articles, but found $filecnt"
  fi
  popd  
fi
