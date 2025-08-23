fp="test_output-disp-0.json"
rm -f $fp
touch $fp
echo "{}" >> $fp

skipper="/home/emily/myrtle/40x120x20w8-32-10_searchSpace.csv"

python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_40x120x20_f64" scyc $fp #$skipper
