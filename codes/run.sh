max=599
for i in `seq 594 $max`
do
    echo "$i"
    chk="checkpoint_$i.pth.tar"
    echo "$chk"
    python -m compressai.utils.update_model -exp exp_08_mse_q8 -a invcompress -checkpoint "$chk"
    python -m compressai.utils.eval_model checkpoint /mnt/6t_hdd/Priyanka/qmap_dataset/kodak -a invcompress -exp exp_08_mse_q8 -s ../results/exp_08 > "$chk.log"

done

