fp="test_output-disp-0.json"
rm -f $fp
touch $fp
echo "{}" >> $fp

#python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_40x120x20_f64" sflt $fp
python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_251x500x600_f64" sflt $fp
#python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_1x400x200_f64" sflt $fp
# new="100x400x200wm-n-k_case1_searchSpace.csv"
# old="100x400x200wm-n-k_case1_searchSpace-before-changes.csv"
# diff $old $new
# new="100x400x200wm-n-k_case1_searchSpace-myrtle-sflt-ranking.csv"
# old="100x400x200wm-n-k_case1_searchSpace-myrtle-sflt-ranking-before-changes.csv"
# diff $old $new
#old="40x120x20wm-n-k_case1_searchSpaceOLD.csv"
#new="40x120x20wm-n-k_searchSpace.csv"
# old="1x400x200wm-n-k_case1_searchSpaceOLD.csv"
# new="1x400x200wm-n-k_searchSpace.csv"
#diff $old $new

#python3 myrtle/myrtle.py "main\$async_dispatch_0_matmul_transpose_b_16x768x768_f64" sflt $fp