#!/bin/bash

cd lp_files
mkdir compressed_lp
mkdir expanded_lp

gcc uncompress_mps.c -o uncompress_mps

for file in $(curl -s http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/ |
                  grep href |
                  sed 's/.*href="//' |
                  sed 's/".*//' |
                  grep '^[a-zA-Z].*'); do
    curl http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/$file > compressed_lp/$file
    gzip -d compressed_lp/$file
done

cat lp_names | while read lp
        do
                echo $lp
                curl https://netlib.org/lp/data/$lp > compressed_lp/$lp
done

for f in compressed_lp/*
    do
        echo $f
        ./uncompress_mps $f > expanded_lp/$(basename $f).mps
done
rmdir -R compressed_lp