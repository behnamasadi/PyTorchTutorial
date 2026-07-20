run(){ docker exec imc python run_loftr.py "$@" 2>&1 | grep -E "\[RESULT\]|Error|Traceback" | grep -vE "Warning"; }
echo "==== LoFTR ===="
run data/extracted/test/stairs work/stairs_loftr --max_dim 840 --tag "stairs_loftr"
run data/extracted/train/imc2023_haiper work/haiper_loftr --max_dim 840 --score_train imc2023_haiper --tag "haiper_loftr"
echo "==== DONE ===="
