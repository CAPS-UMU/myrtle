FOLDER="sensitivity-analysis/manual-c-backend"
# python svr-graph.py "$FOLDER/review-cube.csv" "decFeatures/Cube256x256x256-svr" "decFeatures-256-cube-svr" "decFeatures.txt" "256x256x256"
# python svr-graph.py "$FOLDER/review-distillbert.csv" "decFeatures/Cube512x512x512-svr" "decFeatures-256-cube-svr" "decFeatures.txt" "512x512x512"
# python svr-graph.py "$FOLDER/review-phenomizer.csv" "decFeatures/Phonemizer384x768x768-cube-svr" "decFeatures-256-cube-svr" "decFeatures.txt" "384x768x768"
# python svr-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "decFeatures/Cube128x128x128-svr" "decFeatures-256-cube-svr" "decFeatures.txt" "128x128x128"
# python svr-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "decFeatures/Cube384x384x384-svr" "decFeatures-256-cube-svr" "decFeatures.txt" "384x384x384"

# # "what do you think?"" graph script
# python svr-graph-what-do-you-think.py "$FOLDER/review-cube.csv" "decFeatures/Cube256x256x256-svr-what-do-you-think" "decFeatures-256-cube-svr" "decFeatures.txt"

# # in-progress experiments
# python svr-graph.py "$FOLDER/review-cube.csv" "janFeatures/Cube256x256x256-svr" "janFeatures-256-cube-svr" "janFeatures.txt" "256x256x256"
# python svr-graph.py "$FOLDER/review-distillbert.csv" "janFeatures/Cube512x512x512-svr" "janFeatures-256-cube-svr" "janFeatures.txt" "512x512x512"
# python svr-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "janFeatures/Cube128x128x128-svr" "janFeatures-256-cube-svr" "janFeatures.txt" "128x128x128"
# python svr-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "janFeatures/Cube384x384x384-svr" "janFeatures-256-cube-svr" "janFeatures.txt" "384x384x384"
python approximateC1C2.py "$FOLDER/review-cube.csv" "c1-c2.csv"

python svr-graph.py "$FOLDER/review-cube.csv" "febFeatures/Cube256x256x256-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "256x256x256" ""
python svr-graph.py "$FOLDER/512x512x512wm-n-k_distillbert-results.csv" "febFeatures/Cube512x512x512-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "512x512x512" ""
pwd
python svr-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "febFeatures/Cube128x128x128-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x128" "$FOLDER/128x128x128wm-n-k_searchSpace_c_analyzed-untimed.csv"
python svr-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/Cube384x384x384-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x384x384" "$FOLDER/384x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python svr-graph.py "$FOLDER/review-phenomizer.csv" "febFeatures/Phonemizer384x768x768-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x768x768" ""
# python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/192x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "192x384x384" ""
# python svr-graph.py "$FOLDER/128x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x384x384" ""
# python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x128x64wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x64" ""
# python svr-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x64x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x64x128" ""

# python svr-graph.py "$FOLDER/512x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/512x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "512x384x384" ""
# python svr-graph.py "$FOLDER/32x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/32x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "32x128x128" ""
# python svr-graph.py "$FOLDER/64x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/64x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "64x128x128" ""
python svr-graph.py "$FOLDER/96x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/96x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "96x128x128" "$FOLDER/96x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"

# python svr-graph.py "$FOLDER/128x32x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x32x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x32x128" ""
# python svr-graph.py "$FOLDER/128x64x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x64x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x64x128" ""
# python svr-graph.py "$FOLDER/128x96x128wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x96x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x96x128" ""
# python svr-graph.py "$FOLDER/128x128x32wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x128x32wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x32" ""

# python svr-graph.py "$FOLDER/128x128x64wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x128x64wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x64" ""
# python svr-graph.py "$FOLDER/128x128x96wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/128x128x96wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x96" ""