FOLDER="sensitivity-analysis/manual-c-backend"

python approximateC1C2.py "$FOLDER/review-cube.csv" "c1-c2.csv"

python svr-graph.py "$FOLDER/review-cube.csv" "march12/Cube256x256x256-svr" "march12-256-cube-svr" "march12.txt" "256x256x256" ""
python svr-graph.py "$FOLDER/512x512x512wm-n-k_distillbert-results.csv" "march12/Cube512x512x512-svr" "march12-256-cube-svr" "march12.txt" "512x512x512" ""
pwd
python svr-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "march12/Cube128x128x128-svr" "march12-256-cube-svr" "march12.txt" "128x128x128" "$FOLDER/128x128x128wm-n-k_searchSpace_c_analyzed-untimed.csv"
python svr-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/Cube384x384x384-svr" "march12-256-cube-svr" "march12.txt" "384x384x384" "$FOLDER/384x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/review-phenomizer.csv" "march12/Phonemizer384x768x768-cube-svr" "march12-256-cube-svr" "march12.txt" "384x768x768" ""
python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/192x384x384wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "192x384x384" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/128x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x384x384wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x384x384" "$FOLDER/128x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x128x64wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x128x64" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x64x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x64x128" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"

python svr-graph.py "$FOLDER/512x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "march12/512x384x384wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "512x384x384" "$FOLDER/512x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/32x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/32x128x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "32x128x128" "$FOLDER/32x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/64x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/64x128x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "64x128x128" "$FOLDER/64x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/96x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/96x128x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "96x128x128" "$FOLDER/96x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"

python svr-graph.py "$FOLDER/128x32x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x32x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x32x128" "$FOLDER/128x32x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/128x64x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x64x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x64x128" "$FOLDER/128x64x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/128x96x128wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x96x128wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x96x128" "$FOLDER/128x96x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/128x128x32wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x128x32wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x128x32" "$FOLDER/128x128x32wm-n-k_searchSpace_c_analyzed_untimed.csv"

python svr-graph.py "$FOLDER/128x128x64wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x128x64wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x128x64" "$FOLDER/128x128x64wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/128x128x96wm-n-k_searchSpace_L1_top_10-results.csv" "march12/128x128x96wm-n-k-cube-svr" "march12-256-cube-svr" "march12.txt" "128x128x96" "$FOLDER/128x128x96wm-n-k_searchSpace_c_analyzed_untimed.csv"