echo "Generating myrtle rankings for dispatches 7, and 8 using mode $1..."
mode="$1"
user="hoppip"
user="emily"

myrtleExec="/home/$user/myrtle/myrtle/myrtle.py"

# from inside segfault_debugging, run
# . rank-debugging-search-spaces.sh svrcyc

rm -f "test_output-disp-7-$mode.json"
rm -f "test_output-disp-8-$mode.json"

echo "{}" > "test_output-disp-7-$mode.json"
echo "{}" > "test_output-disp-8-$mode.json"

disp7Path="/home/$user/myrtle/sensitivity-analysis/holistic-data/1x600x400wm-n-k-timed-debug.csv"
disp8Path="/home/$user/myrtle/sensitivity-analysis/holistic-data/1x600x600wm-n-k-timed-debug.csv" 
disp7PathRankSuffix="1x600x400wm-n-k-timed-debug-myrtle-$mode-ranking.csv"
disp8PathRankSuffix="1x600x600wm-n-k-timed-debug-myrtle-$mode-ranking.csv"    

python3 $myrtleExec "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" $mode "test_output-disp-7-$mode.json" $disp7Path
python3 $myrtleExec "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" $mode "test_output-disp-8-$mode.json" $disp8Path

cp "/home/$user/myrtle/sensitivity-analysis/holistic-data/$disp7PathRankSuffix" $disp7PathRankSuffix
cp "/home/$user/myrtle/sensitivity-analysis/holistic-data/$disp8PathRankSuffix" $disp8PathRankSuffix

rm -f "/home/$user/myrtle/sensitivity-analysis/holistic-data/$disp7PathRankSuffix"
rm -f "/home/$user/myrtle/sensitivity-analysis/holistic-data/$disp8PathRankSuffix"

echo "done."