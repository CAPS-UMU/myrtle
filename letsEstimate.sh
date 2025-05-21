dataDir="./graphing/streamingLoadDetail"
echo "Adding estimated runtime data..."

# case 1
python3 tile_sizes_estimate_cycles.py 1200 400 "$dataDir/dispatch_1_case1_everything.csv" "dispatch_1" 1
python3 tile_sizes_estimate_cycles.py 1200 400 "$dataDir/dispatch_1_case2_everything.csv" "dispatch_1" 2

# case 7
python3 tile_sizes_estimate_cycles.py 600 400 "$dataDir/dispatch_7_case1_everything.csv" "dispatch_7" 1
python3 tile_sizes_estimate_cycles.py 600 400 "$dataDir/dispatch_7_case2_everything.csv" "dispatch_7" 2

# case 8
python3 tile_sizes_estimate_cycles.py 600 600 "$dataDir/dispatch_8_case1_everything.csv" "dispatch_8" 1
python3 tile_sizes_estimate_cycles.py 600 600 "$dataDir/dispatch_8_case2_everything.csv" "dispatch_8" 2

