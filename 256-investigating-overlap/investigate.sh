HERE=$(pwd)
TIMED="/home/hoppip/myrtle/256-investigating-overlap/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
ANN="/home/hoppip/myrtle/sensitivity-analysis/beta=0/spm-reg/untimed/256x256x256wm-n-k_ss_c_rem_div_ana.csv"
FULL=""
WEBPAGE_TITLE="overlap-stall-time-for-selected-points"
HTML_NAME="investigate-overlap-stall"

cd ../scripts
python investigate-overlap-stall.py $TIMED $ANN $WEBPAGE_TITLE $HTML_NAME
cd $HERE
#firefox --new-tab $HERE"../scripts/out/"$HTML_NAME
#firefox --new-tab "../scripts/out"$HTML_NAME
# file:///home/hoppip/myrtle/scripts/investigate-overlap-stall.html

