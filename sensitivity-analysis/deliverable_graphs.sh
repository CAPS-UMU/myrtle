here=$(pwd)
# directory in which to look for graphs
outputDir="$here/holistic-data/graphs"
contextImg="$here/../myrtle/graphing/context2.png"

cd "../myrtle"

disp0="$outputDir/d0-1-400-161-rank-actual-filtered.png"
disp1="$outputDir/d1-1-1200-400-rank-actual-filtered.png"
disp7="$outputDir/d7-1-600-400-rank-actual-filtered.png"
disp8="$outputDir/d8-1-600-600-rank-actual-filtered.png"

# bash graph-holistic.sh "sflt" 10
python3 -m graphing.combine_images_w_title $contextImg $disp0 $disp1 $disp7 $disp8
