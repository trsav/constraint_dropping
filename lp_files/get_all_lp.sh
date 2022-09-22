cd lp_files
gcc uncompress_mps.c -o uncompress_mps

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
