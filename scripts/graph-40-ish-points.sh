DIVISOR_FLDR="../sensitivity-analysis/remainder-vs-divisor/div"
REMAINDER_FLDR="../sensitivity-analysis/remainder-vs-divisor/rem"
BOTH_FLDR="../sensitivity-analysis/remainder-vs-divisor/both"
WEBPAGES=""
WEBPAGES512=""
WEBPAGES384=""
WEBPAGES120x120x60=""
WEBPAGES40ishPoints=""
USER="hoppip"

# bash graph-40-ish-points.sh outer-join-each.input

# compile(){
#     ss="$1"
#     echo -e "\tmany_gemms.sh: COMPILE step"
#     uniquePointRegex='^(([0-9]*)x([0-9]*)x([0-9]*))w([0-9]*)-([0-9]*)-([0-9]*)'
#     for ts in $(grep -oE $uniquePointRegex $ss)
#             do
#             eatNum='^([0-9])([0-9])*'
#             M=$(echo $ts | grep -oE $eatNum)
#             tail=${ts#*x}
#             N=$(echo $tail | grep -oE $eatNum)
#             tail=${tail#*x}
#             K=$(echo $tail | grep -oE $eatNum)
#             tail=${tail#*w}
#             m=$(echo $tail | grep -oE $eatNum)
#             tail=${tail#*-}
#             n=$(echo $tail | grep -oE $eatNum)
#             tail=${tail#*-}
#             k=$(echo $tail | grep -oE $eatNum)
#             buildDir="$experimentDir/"$M"x"$N"x"$K"w"$m"-"$n"-"$k
#             echo -e "\t\t$M $N $K $m $n $k with build directory $buildDir"
#             rm -rf $buildDir 2>/dev/null
#             mkdir $buildDir
#             python $prepareParamsScript $params $M $N $K $m $n $k "$gemmDir"
#             if [[ "$(echo $?)" == "0" ]]; 
#                 then
#                 make DEBUG=ON sw -j
#                 if [[ "$(echo $?)" == "0" ]]; 
#                 then
#                     cp -r "$gemmDir/build" $buildDir
#                     cp $params "$buildDir/params.json"
#                 fi
#             fi
#             done
# }

graph(){
     DIMS="$1"
     # concat the timed data
     REMAINDER="$BOTH_FLDR/timed/20-points-div-rem/"$DIMS"wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
     DIVISOR="$BOTH_FLDR/timed/20-points/$DIMS/"$DIMS"wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
     python concatCSVs.py $REMAINDER $DIVISOR "out/$DIMS-div-rem-40ish-points.csv"
     # concat the analysis data
     REMAINDER_ANALYZED="$BOTH_FLDR/untimed/20-points-div-rem/"$DIMS"wm-n-k_ss_c_rem_div_ana_pruned.csv"
     DIVISOR_ANALYZED="$BOTH_FLDR/untimed/20-points/$DIMS/"$DIMS"wm-n-k_ss_c_rem_div_ana_pruned.csv"
     python concatCSVs.py $REMAINDER_ANALYZED $DIVISOR_ANALYZED "out/$DIMS-div-rem-40ish-points-ann.csv"
     # now graph 120x120x60 data
     ANALYZED="out/$DIMS-div-rem-40ish-points-ann.csv"
     TIMED="out/$DIMS-div-rem-40ish-points.csv"
     WEBPAGE_TITLE="$DIMS-experimental-pruning-40ish-points"
     HTML_NAME="out/$DIMS-experimental-pruning-40ish-points"
     MODE="stallCyclesTimed"
     python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
     WEBPAGES40ishPoints+=" $HTML_NAME.html"
    
}

#graph "120x120x60"

while read -r line
do
    echo "$line"
    graph "$line"
done < "$1"

# concat the timed data
# REMAINDER="$BOTH_FLDR/timed/20-points-div-rem/120x120x60wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# DIVISOR="$BOTH_FLDR/timed/20-points/120x120x60/120x120x60wm-n-k_ss_c_rem_div_ana_pruned-results.csv"
# python concatCSVs.py $REMAINDER $DIVISOR "out/120x120x60-div-rem-40ish-points.csv"
# # concat the analysis data
# REMAINDER_ANALYZED="$BOTH_FLDR/untimed/20-points-div-rem/120x120x60wm-n-k_ss_c_rem_div_ana_pruned.csv"
# DIVISOR_ANALYZED="$BOTH_FLDR/untimed/20-points/120x120x60/120x120x60wm-n-k_ss_c_rem_div_ana_pruned.csv"
# python concatCSVs.py $REMAINDER_ANALYZED $DIVISOR_ANALYZED "out/120x120x60-div-rem-40ish-points-ann.csv"
# # now graph 120x120x60 data
# ANALYZED="out/120x120x60-div-rem-40ish-points-ann.csv"
# TIMED="out/120x120x60-div-rem-40ish-points.csv"
# WEBPAGE_TITLE="120x120x60-experimental-pruning-40ish-points"
# HTML_NAME="out/120x120x60-experimental-pruning-40ish-points"
# MODE="stallCyclesTimed"
# python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
# WEBPAGES40ishPoints+=" $HTML_NAME.html"


INDEX_TITLE="Varied Matmul Dimensions - Both Remainders and Divisors - Experimental Pruning"
INDEX_SUMMARY="Experimental Pruning on 40-ish points timed."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/variedDims" $WEBPAGES40ishPoints tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt
