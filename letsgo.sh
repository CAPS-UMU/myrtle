dataDir="./runtime_data_case_1_and_2"
echo "Expanding runtime data with load and corrected size information"

# case 1
python3 tile_size_generator.py 1200 400 "$dataDir/dispatch_1_case_1_graphing.csv" "dispatch_1" 1
python3 tile_size_generator.py 1200 400 "$dataDir/dispatch_1_case_2_graphing.csv" "dispatch_1" 2

# case 7
python3 tile_size_generator.py 600 400 "$dataDir/dispatch_7_case_1_graphing.csv" "dispatch_7" 1
python3 tile_size_generator.py 600 400 "$dataDir/dispatch_7_case_2_graphing.csv" "dispatch_7" 2

# case 8
python3 tile_size_generator.py 600 600 "$dataDir/dispatch_8_case_1_graphing.csv" "dispatch_8" 1
python3 tile_size_generator.py 600 600 "$dataDir/dispatch_8_case_2_graphing.csv" "dispatch_8" 2

