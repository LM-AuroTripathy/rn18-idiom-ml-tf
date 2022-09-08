# python main.py \
#        --data-dir /models/home/auro/pyt-rn18-notebook/imagewoof2-320 \
#        --checkpoint-path /models/home/auro/bk-idiom-ml-tf/rn18/checkpoint/rn18-best-epoch-43-acc-0.7.hdf5 \
#        --do-oob-eval \
#        --eval-batch-size 16


# python main.py \
#        --data-dir /models/home/auro/pyt-rn18-notebook/imagewoof2-320 \
#        --checkpoint-path /models/home/auro/bk-idiom-ml-tf/rn18/checkpoint/rn18-best-epoch-43-acc-0.7.hdf5 \
#        --do-envise-eval \
#        --eval-batch-size 16



python main.py \
       --data-dir /models/home/auro/pyt-rn18-notebook/imagewoof2-320 \
       --checkpoint-path /models/home/auro/bk-idiom-ml-tf/rn18/checkpoint/rn18-best-epoch-43-acc-0.7.hdf5 \
       --do-tune \
       --lr 1e-3 \
       --epochs 1 \
       --tune-batch-size 16




