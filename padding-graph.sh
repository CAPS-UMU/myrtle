FOLDER="sensitivity-analysis/manual-c-backend"
python approximateC1C2.py "$FOLDER/review-cube.csv" "c1-c2.csv"

python padding-graph.py "$FOLDER/review-cube.csv" "padding-naive/Cube256x256x256-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "256x256x256" ""
python padding-graph.py "$FOLDER/512x512x512wm-n-k_distillbert-results.csv" "padding-naive/Cube512x512x512-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "512x512x512" ""
pwd
python padding-graph.py "$FOLDER/128x128x128wm-n-k_top10_l1.csv" "padding-naive/Cube128x128x128-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x128" "$FOLDER/128x128x128wm-n-k_searchSpace_c_analyzed-untimed.csv"
python padding-graph.py "$FOLDER/384x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/Cube384x384x384-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x384x384" "$FOLDER/384x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/review-phenomizer.csv" "padding-naive/Phonemizer384x768x768-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "384x768x768" ""
python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/192x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "192x384x384" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/128x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x384x384" "$FOLDER/128x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x64wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x64" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/192x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x64x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x64x128" "$FOLDER/192x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"

python padding-graph.py "$FOLDER/512x384x384wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/512x384x384wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "512x384x384" "$FOLDER/512x384x384wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/32x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/32x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "32x128x128" "$FOLDER/32x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/64x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/64x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "64x128x128" "$FOLDER/64x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/96x128x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/96x128x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "96x128x128" "$FOLDER/96x128x128wm-n-k_searchSpace_c_analyzed_untimed.csv"

python padding-graph.py "$FOLDER/128x32x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x32x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x32x128" "$FOLDER/128x32x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/128x64x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x64x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x64x128" "$FOLDER/128x64x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/128x96x128wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x96x128wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x96x128" "$FOLDER/128x96x128wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/128x128x32wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x32wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x32" "$FOLDER/128x128x32wm-n-k_searchSpace_c_analyzed_untimed.csv"

python padding-graph.py "$FOLDER/128x128x64wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x64wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x64" "$FOLDER/128x128x64wm-n-k_searchSpace_c_analyzed_untimed.csv"
python padding-graph.py "$FOLDER/128x128x96wm-n-k_searchSpace_L1_top_10-results.csv" "padding-naive/128x128x96wm-n-k-cube-svr" "febFeatures-256-cube-svr" "febFeatures.txt" "128x128x96" "$FOLDER/128x128x96wm-n-k_searchSpace_c_analyzed_untimed.csv"