BOTH_FLDR="../sensitivity-analysis/remainder-vs-divisor/both"
WEBPAGES40ishPoints=""
USER="emily"

#bash graph-both-data.sh dims-csv-name-line-by-line.input both


graph(){
    stringarray=($1) # dims followed by the the CSV name, INCLUDING .csv extension!
    DIMS="${stringarray[0]}"
    TIMED="${stringarray[1]}"
    ANALYZED="${stringarray[2]}"
    FULL="${stringarray[3]}"
    echo $DIMS" -----vvvvvvvvvvvvvvvvvvvvvvvvv--------"
    WEBPAGE_TITLE="$DIMS-myrtle-pruning"
    HTML_NAME="out/$DIMS-myrtle-pruning"
    MODE="stallCyclesTimed" # legacy
    # now graph the data
    python graph-full-timed-pruned.py $TIMED $ANALYZED $FULL $WEBPAGE_TITLE $HTML_NAME $MODE
    WEBPAGES40ishPoints+=" $HTML_NAME.html"    
}


while read -r line
do
   # echo "$line"
    graph "$line"
done < "$1"



INDEX_TITLE="Both Remainders and Divisors - Experimental Pruning"
INDEX_SUMMARY="Experimental Pruning points timed."
echo $INDEX_TITLE > tempTitle.txt
echo $INDEX_SUMMARY > tempSummary.txt
python generate-html-index.py "./out/$2" $WEBPAGES40ishPoints tempTitle.txt tempSummary.txt
rm -rf tempTitle.txt tempSummary.txt
