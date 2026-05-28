BOTH_FLDR="../sensitivity-analysis/remainder-vs-divisor/both"
WEBPAGES40ishPoints=""
USER="emily"

#bash graph-both-data.sh dims-csv-name-line-by-line.input both


graph(){
    stringarray=($1) # dims followed by the the CSV name, INCLUDING .csv extension!
    CSV="${stringarray[1]}"
    DIMS="${stringarray[0]}"
    CSV="${stringarray[1]}"
    echo $DIMS
    echo $CSV
    ANALYZED="../paper/cached-out/out/$DIMS""wm-n-k_ss_c_rem_div_ana_pruned.csv"
    #/home/emily/myrtle/paper/cached-out/out/30x128x125wm-n-k_ss_c_rem_div_ana_pruned.csv
    # ANALYZED="../myrtle/out/$DIMS""wm-n-k_ss_c_rem_div_ana_pruned.csv"
    TIMED="$BOTH_FLDR/timed/$CSV"
    WEBPAGE_TITLE="$DIMS-experimental-pruning-both-div-rem-points"
    HTML_NAME="out/$DIMS-experimental-pruning-both-div-rem-points"
    MODE="stallCyclesTimed" # legacy
    # now graph the data
    python graph-2-kinds2.py $TIMED $ANALYZED $WEBPAGE_TITLE $HTML_NAME $MODE
    WEBPAGES40ishPoints+=" $HTML_NAME.html"
    
}


while read -r line
do
    echo "$line"
    graph "$line"
done < "$1"



INDEX_TITLE="Both Remainders and Divisors - Experimental Pruning"
INDEX_SUMMARY="Experimental Pruning points timed."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/$2" $WEBPAGES40ishPoints tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt
