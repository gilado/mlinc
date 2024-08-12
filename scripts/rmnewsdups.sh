#!/bin/bash
echo "The news dataset contains duplicates."
echo "This script renames files that contain duplicate data"
echo "However, word2vec program does not depend onduplicate removals."
echo "You may still want to run this script for other purposes."
echo "This script execution time may be a few hours"
read -p "Press any key to continue, ^C to abort "
if [ -d data/news/data ] ; then
  pushd data/news/data
  filetot=870521 # expected file count
  filecnt=$(ls -1 | grep article | wc -l)
  if [ "$filecnt" -eq "$filetot" ] ; then
    echo "Marking duplicates "
    let filetot=$(ls -1 | grep "article.*.txt\$" | wc -l)
    let filecnt=0
    let fileunq=0
    declare -A chksums
    for f in article*.txt ; do
      chksum=$(md5sum "$f" | cut -d ' ' -f 1)
      if [[ -v "chksums[$chksum]" ]]; then
          mv "$f" "${f}.dup"
      else
        chksums[$chksum]="$f"
        let fileunq=fileunq+1
      fi
      let filecnt=filecnt+1
      echo -en "\r$filecnt scanned, $fileunq unique, out of $filetot"
    done
  else
    echo "Dataset incomplete: expected $filetot articles, but found $filecnt"
  fi
  popd
else
  echo "Dataset is missing. First run scripts/getnewsdata.sh"
fi
