DIVISOR_FLDR="../sensitivity-analysis/remainder-vs-divisor/div"
REMAINDER_FLDR="../sensitivity-analysis/remainder-vs-divisor/rem"
WEBPAGES=""
USER="hoppip"
#USER="emily"
#/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/only-time/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
#/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
#/home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/div/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_ana-results.csv"
HTML_NAME="out/cube128"
REMAINDER="/home/$USER/myrtle/sensitivity-analysis/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_ana-results_lessThan_1024.csv"
REMAINDER_UT="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
WEBPAGE_TITLE="128x128x128-No-Redundant-Stores"
DIVISOR_UT="no"
DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS
WEBPAGES+=" $HTML_NAME.html"

DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_ana-results.csv"
HTML_NAME="out/cube128-stall-cycles"
REMAINDER="/home/$USER/myrtle/sensitivity-analysis/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_ana-results_lessThan_1024.csv"
REMAINDER_UT="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
WEBPAGE_TITLE="128x128x128-No-Redundant-Stores"
DIVISOR_UT="bars"
DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS
WEBPAGES+=" $HTML_NAME.html"

DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_ana-results.csv"
HTML_NAME="out/cube128-stall-cycles-updated-ssr-configs"
REMAINDER="/home/$USER/myrtle/sensitivity-analysis/remainder-vs-divisor/rem/timed/128x128x128wm-n-k_ss_c_ana-results_lessThan_1024.csv"
REMAINDER_UT="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
WEBPAGE_TITLE="128x128x128-No-Redundant-Stores"
DIVISOR_UT="bars-and-scatter"
DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
MORE_REMAINDERS="$REMAINDER_FLDR/timed/128x128x128wm-n-k_ss_c_rem_ana_pruned-results_lessThan_2048.csv"
# concat the timed data
python concatCSVs.py $REMAINDER $DIVISOR "out/128x128x128-div-rem.csv"
python concatCSVs.py "out/128x128x128-div-rem.csv" $MORE_REMAINDERS  "out/128x128x128-all-less-than-2048.csv"
# concat the analysis data
python concatCSVs.py $REMAINDER_UT $DIVISOR_ANALYSIS "out/128x128x128-div-rem-ann.csv"
python graph-2-kinds.py "out/128x128x128-all-less-than-2048.csv" "out/128x128x128-div-rem-ann.csv" $WEBPAGE_TITLE $HTML_NAME
WEBPAGES+=" $HTML_NAME.html"

DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_no-redundant_deprecated.csv"
HTML_NAME="out/cube128-kernel-vs-dma"
REMAINDER="$REMAINDER_FLDR/timed/128x128x128wm-n-k_ss_c_ana-results-unskipped-deprecated.csv"
REMAINDER_UT="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
WEBPAGE_TITLE="128x128x128-Fixed-Kernel-Time-Parsing-Bug"
DIVISOR_UT="no"
DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"
BUG_FIX="$REMAINDER_FLDR/timed/fixed-parse-bug/128x128x128wm-n-k_ss_c_ana-results.csv"
python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS $BUG_FIX
WEBPAGES+=" $HTML_NAME.html"

# DIVISOR="$DIVISOR_FLDR/timed/512x512x512wm-n-k_distillbert-results.csv"
HTML_NAME="out/cube512"
# REMAINDER="no"
# REMAINDER_UT="no"
# WEBPAGE_TITLE="512x512x512-Redundant-Stores"
# DIVISOR_UT="no"
# DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/512x512x512wm-n-k_ss_c_ana.csv"
# python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS
WEBPAGES+=" $HTML_NAME.html"

INDEX_TITLE="128 Cube Remainder Tile Results"
INDEX_SUMMARY="In-progress results timing tiled matmul on snitch using remainder tiles."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/128-cube-remainders" $WEBPAGES tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt


