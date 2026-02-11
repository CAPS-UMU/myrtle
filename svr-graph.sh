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

python svr-graph.py "$FOLDER/review-cube.csv" "febFeatures/Cube256x256x256-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "256x256x256"
python svr-graph.py "$FOLDER/512x512x512wm-n-k_distillbert-results.csv" "febFeatures/Cube512x512x512-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "512x512x512"
python svr-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "febFeatures/Cube128x128x128-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x128"
python svr-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "febFeatures/Cube384x384x384-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x384x384"
python svr-graph.py "$FOLDER/review-phenomizer.csv" "febFeatures/Phonemizer384x768x768-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x768x768"
