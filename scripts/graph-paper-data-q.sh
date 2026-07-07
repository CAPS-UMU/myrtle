WEBPAGES=""
USER="emily"

#bash graph-paper-data-q.sh dims-csv-name-line-by-line-paper-q.input


graph(){
    stringarray=($1) # dims followed by the the CSV name, INCLUDING .csv extension!
    DIMS="${stringarray[0]}"
    TIMED="${stringarray[1]}"
    echo $DIMS" -----vvvvvvvvvvvvvvvvvvvvvvvvv--------"
    WEBPAGE_TITLE="$DIMS-myrtle-pruning"
    HTML_NAME="out/$DIMS-myrtle-pruning"
    # now graph the data
    python graph-quidditch.py $TIMED $WEBPAGE_TITLE $HTML_NAME
   # WEBPAGES+=" $HTML_NAME.html"    
}


while read -r line
do
   # echo "$line"
    graph "$line"
done < "$1"



# INDEX_TITLE="Quidditch NsNet2 Dispatches - Experimental Pruning"
# INDEX_SUMMARY="Experimental Pruning points timed."
# echo $INDEX_TITLE > tempTitle.txt
# echo $INDEX_SUMMARY > tempSummary.txt
# python generate-html-index.py "./out/$2" $WEBPAGES tempTitle.txt tempSummary.txt
# rm -rf tempTitle.txt tempSummary.txt
