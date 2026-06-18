USER="emily"
DATAROOT="/home/$USER/myrtle/sensitivity-analysis"
BOTHTIMEDDIR="$DATAROOT/remainder-vs-divisor/both/timed"
BOTHANNDIR="$DATAROOT/remainder-vs-divisor/both/untimed"
FULLDIR="$BOTHANNDIR/full" 
ANNDIR="$BOTHANNDIR/ann-to-min-third-ssr-configs"
BETA1DIR="$DATAROOT/beta=1/"
BETA0DIR="$DATAROOT/beta=0/"
#/home/hoppip/myrtle/sensitivity-analysis/beta=1/spm-reg/timed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results-first-200.csv

# graph each
# 128 cube beta = 0
# wc -l "$BOTHTIMEDDIR/128x128x128-bertTiny-results.csv"
# wc -l "$BETA0DIR/spm-reg/timed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results-208-reg-spm.csv"
# python concatCSVs.py "$BOTHTIMEDDIR/128x128x128-bertTiny-results.csv" "$BETA0DIR/spm-reg/timed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results-208-reg-spm.csv" "out/128x128x128-reg-SPM-results.csv"
# wc -l "out/128x128x128-reg-SPM-results.csv"
#/home/hoppip/myrtle/sensitivity-analysis/beta=0/spm-reg/timed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results-200-reg-spm.csv

TIMED="$BETA0DIR/spm-reg/timed/128x128x128-reg-SPM-results.csv"
ANN="$BETA0DIR/spm-reg/untimed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$BETA0DIR/spm-reg/untimed/128x128x128wm-n-k_ss_c_rem_div.csv"
BETA="0"
echo "128x128x128" $TIMED $ANN $FULL $BETA > "dims-csv-name-line-by-line-paper.input"
# 128 cube beta = 1
TIMED="$BETA1DIR/spm-reg/timed/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results-first-200.csv"
ANN="$ANNDIR/128x128x128wm-n-k_ss_c_rem_div_ana_pruned.csv"
FULL="$FULLDIR/128x128x128wm-n-k_ss_c_rem_div.csv"
BETA="1"
echo "128x128x128" $TIMED $ANN $FULL $BETA >> "dims-csv-name-line-by-line-paper.input"

bash graph-exp-data-beta.sh dims-csv-name-line-by-line-paper.input beta1