echo "graph-holistic.sh: ATTN: ONLY RUN this script INSIDE myrtle/sensitvity-analyis/"
echo "graph-holistic.sh: Generating graphs for dispatches 0, 1, 7, and 8 using mode $1..."
# Example run:
# clear;. graph-holistic.sh "sflt" -1
here=$(pwd)
# directory in which to look for graphs
dir="$here/holistic-data"
outputDir="$here/holistic-data/graphs"
mode="$1"
topX="$2"

# python3 -m graphing.coral $dir $mode

cd "../myrtle"

# generate mode-agnostic graphs first
# fp="$dir/1x400x161wm-n-k-padding-k-graphing-logistics.csv" # dispatch 0
# python3 -m graphing.kernelAndTotalTime $fp $outputDir $mode
# fp="$dir/1x1200x400wm-n-k-graphing-logistics.csv"          # dispatch 1
# python3 -m graphing.kernelAndTotalTime $fp $outputDir $mode
# fp="$dir/1x600x400wm-n-k-graphing-logistics.csv"           # dispatch 7
# python3 -m graphing.kernelAndTotalTime $fp $outputDir $mode
# fp="$dir/1x600x600wm-n-k-graphing-logistics.csv"           # dispatch 8
# python3 -m graphing.kernelAndTotalTime $fp $outputDir $mode

# generate myrtle rankings given search space
rm -f "test_output-disp-0-$mode.json" 2> /dev/null
rm -f "test_output-disp-1-$mode.json" 2> /dev/null
rm -f "test_output-disp-7-$mode.json" 2> /dev/null
rm -f "test_output-disp-8-$mode.json" 2> /dev/null
echo "{}" > "test_output-disp-0-$mode.json"
echo "{}" > "test_output-disp-1-$mode.json"
echo "{}" > "test_output-disp-7-$mode.json"
echo "{}" > "test_output-disp-8-$mode.json"
# fp="$dir/1x400x161wm-n-k-graphing-logistics.csv" # dispatch 0
# python3 myrtle.py "main\$async_dispatch_0_matmul_transpose_b_1x400x161_f64" $mode "test_output-disp-0-$mode.json" $fp
fp="$dir/1x400x161wm-n-k-padding-k-graphing-logistics.csv" # dispatch 0 (padded)
python3 myrtle.py "main\$async_dispatch_0_matmul_transpose_b_1x400x161_f64" $mode "test_output-disp-0-$mode.json" $fp
fp="$dir/1x1200x400wm-n-k-graphing-logistics.csv"          # dispatch 1
python3 myrtle.py "main\$async_dispatch_1_matmul_transpose_b_1x1200x400_f64" $mode "test_output-disp-1-$mode.json" $fp
fp="$dir/1x600x400wm-n-k-graphing-logistics.csv"           # dispatch 7
python3 myrtle.py "main\$async_dispatch_7_matmul_transpose_b_1x600x400_f64" $mode "test_output-disp-7-$mode.json" $fp
fp="$dir/1x600x600wm-n-k-graphing-logistics.csv"           # dispatch 8
python3 myrtle.py "main\$async_dispatch_8_matmul_transpose_b_1x600x600_f64" $mode "test_output-disp-8-$mode.json" $fp

# generate graph actual vs predicted for Dispatch Time
fp="$dir/1x400x161wm-n-k-graphing-logistics.csv" # dispatch 0
fp2="1x400x161wm-n-k_searchSpace_analyzed-myrtle-$mode-ranking.csv"
python3 -m graphing.actualVsPredicted $fp $fp2 $outputDir $mode $topX
python3 -m graphing.xVsYs $fp $fp2 $outputDir $mode $topX

# fp="$dir/1x400x161wm-n-k-padding-k-graphing-logistics.csv" # dispatch 0 (padded)
# fp2="1x400x161wm-n-k_searchSpace_analyzed-myrtle-$mode-ranking.csv"
# python3 -m graphing.actualVsPredicted $fp $fp2 $outputDir $mode $topX
# python3 -m graphing.xVsYs $fp $fp2 $outputDir $mode $topX

fp="$dir/1x1200x400wm-n-k-graphing-logistics.csv"          # dispatch 1
fp2="1x1200x400wm-n-k_searchSpace_analyzed-myrtle-$mode-ranking.csv" 
python3 -m graphing.actualVsPredicted $fp $fp2 $outputDir $mode $topX
python3 -m graphing.xVsYs $fp $fp2 $outputDir $mode $topX

fp="$dir/1x600x400wm-n-k-graphing-logistics.csv"           # dispatch 7
fp2="1x600x400wm-n-k_searchSpace_analyzed-myrtle-$mode-ranking.csv" 
python3 -m graphing.actualVsPredicted $fp $fp2 $outputDir $mode $topX
python3 -m graphing.xVsYs $fp $fp2 $outputDir $mode $topX

fp="$dir/1x600x600wm-n-k-graphing-logistics.csv"           # dispatch 8
fp2="1x600x600wm-n-k_searchSpace_analyzed-myrtle-$mode-ranking.csv"  
python3 -m graphing.actualVsPredicted $fp $fp2 $outputDir $mode $topX
python3 -m graphing.xVsYs $fp $fp2 $outputDir $mode $topX

# clean up generated myrtle ranking CSVs
cd $here
rm -f "../myrtle/test_output-disp-0-$mode.json" 
rm -f "../myrtle/test_output-disp-1-$mode.json" 
rm -f "../myrtle/test_output-disp-7-$mode.json" 
rm -f "../myrtle/test_output-disp-8-$mode.json" 

echo "graph-holistic.sh: Graphs saved in directory $outputDir"