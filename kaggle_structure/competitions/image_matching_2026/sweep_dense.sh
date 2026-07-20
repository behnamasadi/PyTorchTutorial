S=data/extracted/train/stairs
echo "==== LoFTR vs RoMa on train/stairs (mAA scored) ===="
docker exec imc python run_loftr.py $S work/tstairs_loftr --max_dim 840 --score_train stairs --tag "LoFTR" 2>&1 | grep -E "\[RESULT\]" 
docker exec imc python run_roma.py  $S work/tstairs_roma  --num 2000 --conf_thr 0.7 --score_train stairs --tag "RoMa(conf0.7)" 2>&1 | grep -E "\[RESULT\]"
echo "==== DISK baseline for reference ===="
docker exec imc python experiment.py $S work/tstairs_disk --matchers disk --score_train stairs --tag "DISK" 2>&1 | grep -E "\[RESULT\]"
echo "==== DONE ===="
