set -e
D=data/extracted/train/imc2023_haiper
S=data/extracted/test/stairs
run(){ docker exec imc python experiment.py "$@" 2>&1 | grep -E "\[RESULT\]"; }
echo "==== HAIPER (mAA scored) ===="
run $D work/e1 --matchers disk                    --score_train imc2023_haiper --tag "haiper_disk"
run $D work/e2 --matchers disk --ba               --score_train imc2023_haiper --tag "haiper_disk+ba"
run $D work/e3 --matchers disk --refine_pp        --score_train imc2023_haiper --tag "haiper_disk+pp"
run $D work/e4 --matchers disk --ba --refine_pp   --score_train imc2023_haiper --tag "haiper_disk+ba+pp"
run $D work/e5 --matchers disk,aliked             --score_train imc2023_haiper --tag "haiper_ens"
run $D work/e6 --matchers disk,aliked --ba --refine_pp --score_train imc2023_haiper --tag "haiper_ens+ba+pp"
echo "==== STAIRS (hard, reg only) ===="
run $S work/s1 --matchers disk                    --tag "stairs_disk"
run $S work/s2 --matchers disk,aliked             --tag "stairs_ens"
run $S work/s3 --matchers disk,aliked --ba --refine_pp --tag "stairs_ens+ba+pp"
echo "==== SWEEP DONE ===="
