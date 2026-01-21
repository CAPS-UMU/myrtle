# Train on small cube, run on large cube

## notes

```
/home/hoppip/myrtle/fitLinePlusMoreMetrics.py
python svr-graph.py "review-distillbert.csv" "Cube512x512x512-svr" "256-cube-svr"
  916  python svr-graph.sh
  917  python svr-graph.py "review-cube.csv" "Cube256x256x256-svr" "256-cube-svr"clear
  918  clear
  919  bash svr-graph.sh 
```

## set up

```
source .venv/bin/activate
```



## Recreate Most Recent Results

### by running `bash svr-graph.sh`

which contains runs of script `svr-graph.py`

```
python svr-graph.py "review-cube.csv" "Cube256x256x256-svr" "256-cube-svr"
python svr-graph.py "review-distillbert.csv" "Cube512x512x512-svr" "256-cube-svr" 
python svr-graph.py "review-phenomizer.csv" "Phenomizer384x768x768-svr" "768-phenomizer-svr"
python svr-graph.py "review-distillbert.csv" "Cube512x512x512-svr-phenom" "768-phenomizer-svr"
```

which takes arguments `<test-data.csv> <html-output-filename> <pretrained-SVR-Pickle-File-Name>`

If the pickle file containing the SVR model object does not exist, a new model is trained using `<test-data.csv>` and saved under `<pretrained-SVR-Pickle-File-Name>.pickle`