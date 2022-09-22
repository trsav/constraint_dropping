cd lp_files

cat lp_names | while read lp
do
	echo $lp
	curl https://netlib.org/lp/data/$lp > compressed_lp/$lp
done


for f in compressed_lp/*
    do
	echo $f

	./emps $f > expanded_lp/$(basename $f).mps
done
