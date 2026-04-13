python topTenFromMNK.py "../128x128x128/input.txt" "../128x128x128" _ss_c_rem_ana
python topTenFromMNK.py "../128x128x128/input.txt" "../128x128x128" all

oldDivisors="../128x128x128/before-m-rem-update/128x128x128wm-n-k_ss_c_ana.csv"
oldRemainders="../128x128x128/before-m-rem-update/128x128x128wm-n-k_ss_c_rem_ana.csv"
newDivisors="../128x128x128/128x128x128wm-n-k_ss_c_ana.csv"
newRemainders="../128x128x128/128x128x128wm-n-k_ss_c_rem_ana.csv"

python compareChanges.py $oldDivisors $oldRemainders $newDivisors $newRemainders