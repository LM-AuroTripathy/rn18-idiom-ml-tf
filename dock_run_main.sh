pip install -r requirements.txt

python main.py \
       --data-dir ../imagewoof2-320 \
       --checkpoint-path ../rn18-best-epoch-43-acc-0.7.hdf5 \
       --do-oob-eval \
       --eval-batch-size 16


python main.py \
       --data-dir ../imagewoof2-320 \
       --checkpoint-path ../rn18-best-epoch-43-acc-0.7.hdf5 \
       --do-envise-eval \
       --eval-batch-size 16


python main.py \
       --data-dir ../imagewoof2-320 \
       --checkpoint-path ../rn18-best-epoch-43-acc-0.7.hdf5 \
       --do-tune \
       --lr 1e-5 \
       --epochs 1 \
       --tune-batch-size 16
