USER="hoppip"
#USER="emily"
MYHOME="/home/$USER/"
MYMYRTLE=$MYHOME"myrtle/"
TIMED=$MYMYRTLE"sensitivity-analysis/beta=0/spm-reg/timed/"

# ORIG=$TIMED"128x128x128-reg-SPM-results.csv"
# TOADD=$MYMYRTLE"128x128x128-time-40-40-65/128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# OUTPUT="out/128cube-with-40-40-65.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

# ORIG=$TIMED"128x128x128-reg-SPM-results.csv"
# TOADD=$MYMYRTLE"128x128x128wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# OUTPUT="out/128cube-add-skipped.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

# ORIG=$TIMED"192x384x384-bgeSmall-reg-SPM-results.csv"
# TOADD=$MYMYRTLE"192x384x384wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# OUTPUT="out/192x384x384-add-skipped.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

ORIG=$TIMED"512x512x512wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
TOADD=$MYMYRTLE"512x512x512wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
OUTPUT="out/512x512x512-add-skipped.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT agressive