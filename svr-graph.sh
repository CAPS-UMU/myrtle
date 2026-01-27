python svr-graph.py "review-cube.csv" "decFeatures/Cube256x256x256-svr" "decFeatures-256-cube-svr" "decFeatures.txt"
python svr-graph.py "review-distillbert.csv" "decFeatures/Cube512x512x512-svr" "decFeatures-256-cube-svr" "decFeatures.txt"
python svr-graph.py "review-phenomizer.csv" "decFeatures/Phonemizer384x768x768-cube-svr" "decFeatures-256-cube-svr" "decFeatures.txt"
# "what do you think?"" graph script
python svr-graph-what-do-you-think.py "review-cube.csv" "decFeatures/Cube256x256x256-svr-what-do-you-think" "decFeatures-256-cube-svr" "decFeatures.txt"
# in-progress experiments
# python svr-graph.py "review-cube.csv" "janFeatures/Cube256x256x256-svr" "janFeatures-256-cube-svr" "janFeatures.txt"
# python svr-graph.py "review-distillbert.csv" "janFeatures/Cube512x512x512-svr" "janFeatures-256-cube-svr" "janFeatures.txt"
# python svr-graph.py "review-phenomizer.csv" "janFeatures/Phonemizer384x768x768-cube-svr" "janFeatures-256-cube-svr" "janFeatures.txt"
# python svr-graph.py "review-phenomizer.csv" "Phenomizer384x768x768-svr" "768-phenomizer-svr" # need to debug
# python svr-graph.py "review-distillbert.csv" "Cube512x512x512-svr-phenom" "768-phenomizer-svr" # need to debug