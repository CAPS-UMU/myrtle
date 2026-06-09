DATAROOT="/home/hoppip/myrtle/sensitivity-analysis"
BOTHTIMEDDIR="$DATAROOT/remainder-vs-divisor/both/timed"
BOTHANNDIR="$DATAROOT/remainder-vs-divisor/both/untimed"
FULLDIR="$BOTHANNDIR/full" 
ANNDIR="$BOTHANNDIR/ann-to-min-third-ssr-configs"

# python concatCSVs.py $REMAINDER $DIVISOR "out/128x128x128-div-rem.csv"

# Bert Tiny
# bertlt1024="$DATAROOT/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_ana-results_lessThan_1024.csv"
# bertlt2048="$DATAROOT/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_rem_ana_pruned-results_lessThan_2048.csv"
# bertDivs="$DATAROOT/remainder-vs-divisor/div/timed/128x128x128wm-n-k_ss_c_ana-results.csv"

# # miniLM
# PARTIALRESULTS="$DATAROOT/remainder-vs-divisor/both/timed/384x384x384-div-rem-partial-results.csv"
# MOREDATA="$DATAROOT/remainder-vs-divisor/both/timed/384x384x384wm-n-k_ss_c_rem_ana_pruned-results-more.csv"

# # BGE Small
# bgeTIMEOUT="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pr_sel_sflt-timeout-results.csv"
# bge25="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pr_sel_sflt-results-unskipped-25.csv"
# bge151="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pr_sel_sflt-results-unskipped-151.csv"
# bge65="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pruned-results-unskipped-65.csv"
# bge39="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pruned-39-min-ssr-configs-results.csv"
# bgeCrash="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pruned-before-crash-results-unskipped.csv"
# bge178="$BOTHTIMEDDIR/192x384x384/192x384x384wm-n-k_ss_c_rem_div_ana_pruned-results-unskipped-178.csv"

# # Roberta
# robertaTIMEOUT="$BOTHTIMEDDIR/128x768x768wm-n-k_ss_c_rem_div_ana_pr_sel_sflt-timeout-results.csv"
# robertaREST="$BOTHTIMEDDIR/128x768x768wm-n-k_ss_c_rem_div_ana_pruned-results-unskipped.csv"

# echo "files we need to concat..."
# wc -l $bertlt2048
# wc -l $bertDivs
# python concatCSVs.py $bertlt2048 $bertDivs "out/128x128x128-bertTiny-results.csv"
# wc -l "out/128x128x128-bertTiny-results.csv"
# wc -l $bertlt1024
# python concatCSVs.py $bertlt1024 "out/128x128x128-bertTiny-results.csv" "out/128x128x128-bertTiny-results.csv"
# wc -l "out/128x128x128-bertTiny-results.csv"

# wc -l $PARTIALRESULTS
# wc -l $MOREDATA
# python concatCSVs.py $PARTIALRESULTS $MOREDATA "out/384x384x384-miniLM-results.csv"
# wc -l "out/384x384x384-miniLM-results.csv"

# wc -l $bgeTIMEOUT
# wc -l $bge25
# python concatCSVs.py $bgeTIMEOUT $bge25 "out/192x384x384-bgeSmall-25-timeout-results.csv"
# wc -l "out/192x384x384-bgeSmall-25-timeout-results.csv"
# wc -l $bge65
# python concatCSVs.py "out/192x384x384-bgeSmall-25-timeout-results.csv" $bge65 "out/192x384x384-bgeSmall-25-timeout-65-results.csv"
# wc -l "out/192x384x384-bgeSmall-25-timeout-65-results.csv"
# wc -l $bge151
# python concatCSVs.py "out/192x384x384-bgeSmall-25-timeout-65-results.csv" $bge151 "out/192x384x384-bgeSmall-results.csv"
# wc -l "out/192x384x384-bgeSmall-results.csv"
# wc -l $bge39
# python concatCSVs.py "out/192x384x384-bgeSmall-results.csv" $bge39 "out/192x384x384-bgeSmall-results.csv"
# wc -l "out/192x384x384-bgeSmall-results.csv"
# wc -l $bgeCrash
# python concatCSVs.py "out/192x384x384-bgeSmall-results.csv" $bgeCrash "out/192x384x384-bgeSmall-results.csv"
# wc -l "out/192x384x384-bgeSmall-results.csv"
# wc -l $bge178
# python concatCSVs.py "out/192x384x384-bgeSmall-results.csv" $bge178 "out/192x384x384-bgeSmall-results.csv"
# wc -l "out/192x384x384-bgeSmall-results.csv"

# wc -l $robertaTIMEOUT
# wc -l $robertaREST
# python concatCSVs.py $robertaTIMEOUT $robertaREST "out/128x768x768-roberta-results.csv"
# wc -l "out/128x768x768-roberta-results.csv"

# graph each

miniLMTIMED="$BOTHTIMEDDIR/384x384x384-miniLM-results.csv"
#python createTimeoutRows.py "$miniLMTIMED" 8311517
miniLMANN="$ANNDIR/384x384x384wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/384x384x384wm-n-k_ss_c_rem_div.csv"
echo "384x384x384 $miniLMTIMED $miniLMANN" $FULL > "dims-csv-name-line-by-line-paper.input"

bertTinyTIMED="$BOTHTIMEDDIR/128x128x128-bertTiny-results.csv"
bertTinyANN="$ANNDIR/128x128x128wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/128x128x128wm-n-k_ss_c_rem_div.csv"
echo "128x128x128 $bertTinyTIMED $bertTinyANN" $FULL >> "dims-csv-name-line-by-line-paper.input"

bertMiniTIMED="$BOTHTIMEDDIR/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
bertMiniANN="$ANNDIR/256x256x256wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/256x256x256wm-n-k_ss_c_rem_div.csv"
echo "256x256x256 $bertMiniTIMED $bertMiniANN" $FULL >> "dims-csv-name-line-by-line-paper.input"

bgeSmallTIMED="$BOTHTIMEDDIR/192x384x384-bgeSmall-results.csv"
#python createTimeoutRows.py "$bgeSmallTIMED" 4259722
bgeSmallANN="$ANNDIR/192x384x384wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/192x384x384wm-n-k_ss_c_rem_div.csv"
echo "192x384x384 $bgeSmallTIMED $bgeSmallANN" $FULL >> "dims-csv-name-line-by-line-paper.input"

robertaTIMED="$BOTHTIMEDDIR/128x768x768-roberta-results.csv"
#python createTimeoutRows.py "$robertaTIMED" 11391161
robertaANN="$ANNDIR/128x768x768wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/128x768x768wm-n-k_ss_c_rem_div.csv"
echo "128x768x768 $robertaTIMED $robertaANN" $FULL >> "dims-csv-name-line-by-line-paper.input"

TIMED="$BOTHTIMEDDIR/512x512x512wm-n-k_ss_c_rem_div_ana_pruned-results-include-timeout.csv"
#python createTimeoutRows.py "$TIMED" 19921156
ANN="$ANNDIR/512x512x512wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/512x512x512wm-n-k_ss_c_rem_div.csv"
echo "512x512x512" $TIMED $ANN $FULL >> "dims-csv-name-line-by-line-paper.input"

# bash graph-paper-data.sh dims-csv-name-line-by-line-paper.input transformers
bash graph-paper-data-q.sh dims-csv-name-line-by-line-paper-q.input q