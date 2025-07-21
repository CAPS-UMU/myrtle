echo "Generating myrtle rankings for dispatches 1, 7, and 8 using mode $1..."
mode="$1"
user="hoppip"
user="emily"
here=$(pwd)
echo $here
cd "../../"

pwd 
python3 -m graphing.deliverableGraphs /home/$user/myrtle/myrtle/accuracy/with-old-data $mode
cd $here

# user="emily"
# rm -f "test_output-disp-1-$mode.json"
# rm -f "test_output-disp-7-$mode.json"
# rm -f "test_output-disp-8-$mode.json"

# echo "{}" > "test_output-disp-1-$mode.json"
# echo "{}" > "test_output-disp-7-$mode.json"
# echo "{}" > "test_output-disp-8-$mode.json"

# python3 /home/$user/myrtle/myrtle/myrtle.py "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64" $mode "test_output-disp-1-$mode.json" /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_1_case1_everything.csv
# python3 /home/$user/myrtle/myrtle/myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" $mode "test_output-disp-7-$mode.json" /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything.csv
# python3 /home/$user/myrtle/myrtle/myrtle.py "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" $mode "test_output-disp-8-$mode.json" /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything.csv

# cp /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_1_case1_everything-myrtle-$mode-ranking.csv dispatch_1_case1_everything-myrtle-$mode-ranking.csv
# cp /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything-myrtle-$mode-ranking.csv dispatch_7_case1_everything-myrtle-$mode-ranking.csv
# cp /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything-myrtle-$mode-ranking.csv dispatch_8_case1_everything-myrtle-$mode-ranking.csv

# rm -f /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_1_case1_everything-myrtle-$mode-ranking.csv
# rm -f /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_7_case1_everything-myrtle-$mode-ranking.csv
# rm -f /home/$user/myrtle/sensitivity-analysis/holistic-data/dispatch_8_case1_everything-myrtle-$mode-ranking.csv

# echo "done."