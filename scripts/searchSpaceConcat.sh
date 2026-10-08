# USER="hoppip"
USER="emily"
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

# ORIG=$TIMED"512x512x512wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
# TOADD=$MYMYRTLE"512x512x512wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# OUTPUT="out/512x512x512-add-skipped.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

# ORIG=$MYMYRTLE"256-investigating-overlap/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# TOADD=$MYMYRTLE"256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results-stall-cycles.csv"
# OUTPUT="out/256-stall-cycles-morefmadds.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT polite

# ORIG=$TIMED"256x256x256wm-n-k_ss_c_rem_div_ana-results-dma-only.csv"
# TOADD=$MYMYRTLE"256x256x256wm-n-k_ss_c_rem_div_ana-results.csv"
# OUTPUT="out/256-morefmadds.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT polite

# ORIG=$TIMED"384x384x384wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
# TOADD=$MYMYRTLE"384x384x384wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# OUTPUT="out/384-no-point-left-behind.csv"
# python concatCSVs.py $ORIG $TOADD $OUTPUT polite

# ORIG="/home/emily/myrtle/256-investigating-overlap/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# TOADD="/home/emily/myrtle/256-investigating-overlap/256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results-stall-cycles.csv"
# OUTPUT="out/256-overlap-stall-experiments"
# python concatCSVs.py $ORIG $TOADD $OUTPUT polite
TENK="/home/emily/myrtle/10k-12k-data/"
ORIG=$TIMED"256x256x256wm-n-k_ss_c_rem_div_ana-results-dma-only.csv"
TOADD=$TENK"256x256x256wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
OUTPUT="out/256-no-point-left-behind.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT polite

ORIG=$TIMED"128x768x768-roberta-reg-SPM-results.csv"
TOADD=$TENK"128x768x768wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
OUTPUT="out/128x768x768-no-point-left-behind.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

ORIG=$TIMED"192x384x384-bgeSmall-reg-SPM-results.csv"
TOADD=$TENK"192x384x384wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
OUTPUT="out/192x384x384-no-point-left-behind.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT agressive

ORIG=$TIMED"384x384x384wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
TOADD=$TENK"384x384x384wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
OUTPUT="out/384-no-point-left-behind.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT polite

ORIG=$TIMED"512x512x512wm-n-k_rem_div_ana_pruned-results-dma-only.csv"
TOADD=$TENK"512x512x512wm-n-k_ss_c_rem_div_ana_pruned-results-unskipped.csv"
OUTPUT="out/512-no-point-left-behind.csv"
python concatCSVs.py $ORIG $TOADD $OUTPUT agressive