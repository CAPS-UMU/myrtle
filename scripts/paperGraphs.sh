# python concatCSVs.py $REMAINDER $DIVISOR "out/128x128x128-div-rem.csv"

DATAROOT="/home/emily/myrtle/sensitivity-analysis"
BOTHTIMEDDIR="$DATAROOT/remainder-vs-divisor/both/timed"
BOTHANNDIR="$DATAROOT/remainder-vs-divisor/both/untimed"

# Bert Tiny
bertlt1024="$DATAROOT/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_ana-results_lessThan_1024.csv"
bertlt2048="$DATAROOT/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_rem_ana_pruned-results_lessThan_2048.csv"
bertDivs="$DATAROOT/remainder-vs-divisor/div/timed/128x128x128wm-n-k_ss_c_ana-results.csv"

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

# create list of timed and annotated points to graph 
# annotated should include the timed points + untimed points, only w/ analysis info
TIMED=""
ANNED=""

miniLMTIMED="$BOTHTIMEDDIR/384x384x384-miniLM-results.csv"
TIMED+=" $miniLMTIMED"
miniLMANN="$BOTHANNDIR/384x384x384wm-n-k_ss_c_rem_div_ana_pruned.csv"
ANNED+=" $miniLMANN"
echo "384x384x384 $miniLMTIMED $miniLMANN" > "dims-csv-name-line-by-line-paper.input"

bertTinyTIMED="$BOTHTIMEDDIR/128x128x128-bertTiny-results.csv"
TIMED+=" $bertTinyTIMED"
bertTinyANN="$BOTHANNDIR/128x128x128-div-rem-ann.csv"
ANNED+=" $bertTinyANN"
echo "128x128x128 $bertTinyTIMED $bertTinyANN" >> "dims-csv-name-line-by-line-paper.input"

bertMiniTIMED="$BOTHTIMEDDIR/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
TIMED+=" $bertMiniTIMED"
bertMiniANN="$BOTHANNDIR/256x256x256wm-n-k_ss_c_rem_div_ana_pruned.csv"
ANNED+=" $bertMiniANN"
echo "256x256x256 $bertMiniTIMED $bertMiniANN" >> "dims-csv-name-line-by-line-paper.input"

bgeSmallTIMED="$BOTHTIMEDDIR/192x384x384-bgeSmall-results.csv"
TIMED+=" $bgeSmallTIMED"
bgeSmallANN="$BOTHANNDIR/192x384x384wm-n-k_ss_c_rem_div_ana_pruned.csv"
ANNED+=" $bgeSmallANN"
echo "192x384x384 $bgeSmallTIMED $bgeSmallANN" >> "dims-csv-name-line-by-line-paper.input"

# robertaTIMED="$BOTHTIMEDDIR/128x768x768-roberta-results.csv"
# TIMED+=" $robertaTIMED"
# robertaANN="$BOTHANNDIR/128x768x768wm-n-k_ss_c_rem_div_ana_pruned.csv"
# ANNED+=" $robertaANN"
# echo "128x768x768 $robertaTIMED $robertaANN" >> "dims-csv-name-line-by-line-paper.input"

# echo $TIMED
# ls $TIMED
# echo $ANNED


bash graph-paper-data.sh dims-csv-name-line-by-line-paper.input both