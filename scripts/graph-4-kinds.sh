DIVISOR_FLDR="../sensitivity-analysis/remainder-vs-divisor/div"
REMAINDER_FLDR="../sensitivity-analysis/remainder-vs-divisor/rem"
BOTH_FLDR="../sensitivity-analysis/remainder-vs-divisor/both"
WEBPAGES=""
WEBPAGES512=""
WEBPAGES384=""
WEBPAGES120x120x60=""
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
HTML_NAME="out/cube128-stall-cycles-many-many-graphs"
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
# python graph-2-kinds.py "out/128x128x128-all-less-than-2048.csv" "out/128x128x128-div-rem-ann.csv" $WEBPAGE_TITLE $HTML_NAME
WEBPAGES+=" $HTML_NAME.html"

# graph same data again, but fewer graphs
WEBPAGE_TITLE="128x128x128-No-Redundant-Stores"
HTML_NAME="out/cube128-stall-cycles-experimental-pruning"
python graph-2-kinds.py "out/128x128x128-all-less-than-2048.csv" "out/128x128x128-div-rem-ann.csv" $WEBPAGE_TITLE $HTML_NAME
WEBPAGES+=" $HTML_NAME.html"

# graph 512 data with experimental pruning methods
DIVISOR_ANALYSIS="/home/$USER/myrtle/myrtle/out/512x512x512wm-n-k_ss_c_ana.csv"
DIVISOR="$DIVISOR_FLDR/timed/512x512x512wm-n-k_distillbert-results-removed-missing.csv"
WEBPAGE_TITLE="512-cube-divisor-tiles-only-experimental-pruning"
HTML_NAME="out/cube512-divisors-experimental-pruning"
MODE="noStallCyclesTimed"
python graph-2-kinds2.py $DIVISOR $DIVISOR_ANALYSIS $WEBPAGE_TITLE $HTML_NAME $MODE
WEBPAGES512+=" $HTML_NAME.html"

# graph 384 data with experimental pruning methods
ANALYZED="/home/$USER/myrtle/myrtle/out/384x384x384wm-n-k_ss_c_rem_ana.csv"
TIMED="$REMAINDER_FLDR/timed/384x384x384wm-n-k-partial-results.csv"
WEBPAGE_TITLE="cube384-remainders-only-experimental-pruning"
HTML_NAME="out/cube384-remainders-only-experimental-pruning"
MODE="stallCyclesTimed"
python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
WEBPAGES384+=" $HTML_NAME.html"

# graph 120x120x60 data with experimental pruning methods
ANALYZED="$BOTH_FLDR/untimed/20-points/120x120x60/120x120x60wm-n-k_ss_c_rem_div_ana_pruned.csv"
TIMED="$BOTH_FLDR/timed/20-points/120x120x60/120x120x60wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
WEBPAGE_TITLE="120x120x60-experimental-pruning-20-points"
HTML_NAME="out/120x120x60-experimental-pruning-20-points"
MODE="stallCyclesTimed"
python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
WEBPAGES120x120x60+=" $HTML_NAME.html"

# concat the timed data
REMAINDER="$REMAINDER_FLDR/timed/384x384x384wm-n-k-partial-results.csv"
DIVISOR="$DIVISOR_FLDR/timed/384x384x384wm-n-k-partial-results.csv"
python concatCSVs.py $REMAINDER $DIVISOR "out/384x384x384-div-rem.csv"
# concat the analysis data
REMAINDER_ANALYZED="/home/$USER/myrtle/myrtle/out/384x384x384wm-n-k_ss_c_rem_ana.csv"
DIVISOR_ANALYZED="/home/$USER/myrtle/myrtle/out/384x384x384wm-n-k_ss_c_ana.csv"
python concatCSVs.py $REMAINDER_ANALYZED $DIVISOR_ANALYZED "out/384x384x384-div-rem-ann.csv"
# now graph 384 data
ANALYZED="out/384x384x384-div-rem-ann.csv"
TIMED="out/384x384x384-div-rem.csv"
WEBPAGE_TITLE="384-cube-remainder-and-divisor-tiles-experimental-pruning"
HTML_NAME="out/cube384-remainders-and-divisors-experimental-pruning"
MODE="stallCyclesTimed"
python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
WEBPAGES384+=" $HTML_NAME.html"


HTML_NAME="out/cube128-kernel-vs-dma"
DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_no-redundant_deprecated.csv"
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
WEBPAGES512+=" $HTML_NAME.html"

INDEX_TITLE="128 Cube Remainder Tile Results"
INDEX_SUMMARY="In-progress results timing tiled matmul on snitch using remainder tiles."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/128-cube-remainders" $WEBPAGES tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt

INDEX_TITLE="512 Cube Divisor Tiles - Experimental Pruning"
INDEX_SUMMARY="In-progress results timing tiled matmul (no remainder tiles) on snitch."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/512-cube-divisors" $WEBPAGES512 tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt

INDEX_TITLE="384 Cube Remainder Tiles Only - Experimental Pruning"
INDEX_SUMMARY="In-progress results timing tiled matmul (no remainder tiles) on snitch."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/384-remainders-only" $WEBPAGES384 tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt

INDEX_TITLE="120x120x60 Both Remainders and Divisors- Experimental Pruning"
INDEX_SUMMARY="Experimental Pruning on 20 points timed."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/120x120x60" $WEBPAGES120x120x60 tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt
