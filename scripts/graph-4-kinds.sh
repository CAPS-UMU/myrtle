DIVISOR_FLDR="../sensitivity-analysis/remainder-vs-divisor/div"
REMAINDER_FLDR="../sensitivity-analysis/remainder-vs-divisor/rem"

#/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/only-time/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
#/home/hoppip/myrtle/sensitivity-analysis/redundant-vs-no-redundant-stores/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
#/home/hoppip/myrtle/sensitivity-analysis/remainder-vs-divisor/div/128x128x128wm-n-k_ss_c_ana-results-no-redundant.csv
DIVISOR="$DIVISOR_FLDR/timed/128x128x128wm-n-k_ss_c_no-redundant.csv"
HTML_NAME="out/cube128"
REMAINDER="no"
REMAINDER_UT="/home/hoppip/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_rem_ana.csv"
WEBPAGE_TITLE="128x128x128-No-Redundant-Stores"
DIVISOR_UT="no"
# DIVISOR_ANALYSIS="$DIVISOR_FLDR/analysis/128x128x128wm-n-k_ss_c_ana.csv"
DIVISOR_ANALYSIS="/home/hoppip/myrtle/myrtle/out/128x128x128wm-n-k_ss_c_ana.csv"

python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS

DIVISOR="$DIVISOR_FLDR/timed/512x512x512wm-n-k_distillbert-results.csv"
HTML_NAME="out/cube512"
REMAINDER="no"
REMAINDER_UT="no"
WEBPAGE_TITLE="512x512x512-Redundant-Stores"
DIVISOR_UT="no"
#DIVISOR_ANALYSIS="$DIVISOR_FLDR/analysis/512x512x512wm-n-k_ss_c_ana.csv"
DIVISOR_ANALYSIS="/home/hoppip/myrtle/myrtle/out/512x512x512wm-n-k_ss_c_ana.csv"

python graph-4-kinds.py $DIVISOR $HTML_NAME $REMAINDER $REMAINDER_UT $WEBPAGE_TITLE $DIVISOR_UT $DIVISOR_ANALYSIS

#python padding-graph.py "$FOLDER/512x512x512wm-n-k_distillbert-results.csv" "padding-naive/Cube512x512x512-svr" "" "" "512x512x512" ""

# python padding-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "padding-naive/Cube128x128x128-svr" "$FOLDER/128x128x128-padded-128x144x128wm-n-k_searchSpace_some-padded-results.csv" "" "128x128x128" "$FOLDER/128x128x128wm-n-k_searchSpace_c_analyzed-untimed.csv"
# python padding-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/Cube384x384x384-svr" "" "" "384x384x384" "$FOLDER/384x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/review-phenomizer.csv" "padding-naive/Phonemizer384x768x768-cube-svr" "" "" "384x768x768" ""
# python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/192x384x384wm-n-k-cube-svr" "" "" "192x384x384" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/128x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x384x384wm-n-k-cube-svr" "" "" "128x384x384" "$FOLDER/128x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x64wm-n-k-cube-svr" "" "" "128x128x64" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x64x128wm-n-k-cube-svr" "" "" "128x64x128" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"

# python padding-graph.py "$FOLDER/512x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/512x384x384wm-n-k-cube-svr" "" "" "512x384x384" "$FOLDER/512x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/32x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/32x128x128wm-n-k-cube-svr" "" "" "32x128x128" "$FOLDER/32x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/64x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/64x128x128wm-n-k-cube-svr" "" "" "64x128x128" "$FOLDER/64x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/96x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/96x128x128wm-n-k-cube-svr" "" "" "96x128x128" "$FOLDER/96x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"

# python padding-graph.py "$FOLDER/128x32x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x32x128wm-n-k-cube-svr" "" "" "128x32x128" "$FOLDER/128x32x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/128x64x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x64x128wm-n-k-cube-svr" "" "" "128x64x128" "$FOLDER/128x64x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/128x96x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x96x128wm-n-k-cube-svr" "" "" "128x96x128" "$FOLDER/128x96x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/128x128x32wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x32wm-n-k-cube-svr" "" "" "128x128x32" "$FOLDER/128x128x32wm-n-k_searchSpace_c_analyzed_untimed.csv"

# python padding-graph.py "$FOLDER/128x128x64wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x64wm-n-k-cube-svr" "" "" "128x128x64" "$FOLDER/128x128x64wm-n-k_searchSpace_c_analyzed_untimed.csv"
# python padding-graph.py "$FOLDER/128x128x96wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x96wm-n-k-cube-svr" "" "" "128x128x96" "$FOLDER/128x128x96wm-n-k_searchSpace_c_analyzed_untimed.csv"