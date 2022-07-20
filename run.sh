# TRAIN
# SATRN debug
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=36002 ./tools/train.py ./exps/satrn_debug.py --launcher=pytorch --work-dir=satrn_debug
# SATRN original train
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=36002 ./tools/train.py ./exps/satrn.py --launcher=pytorch --work-dir=wd_satrn
# SATRN change learning rate only
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=36002 ./tools/train.py ./exps/satrn_debug.py --launcher=pytorch --work-dir=wd_satrn_dbg
# SATRN change learning rate only
python -m torch.distributed.launch --nproc_per_node=8 --master_port=36002 ./tools/train.py ./exps/satrn_ST_MJ_qd.py --launcher=pytorch


# TEST
python -m torch.distributed.launch --nproc_per_node=4 --master_port=27271 ./tools/test.py ./exps/satrn_ST_qd.py ./work_dirs/satrn_ST_qd/epoch_6.pth --eval=acc --launcher=pytorch

# CLEAN
sleep 5
kill $(ps aux | grep "zctian/anaconda3/envs/mmlab" | grep -v grep | awk '{print $2}')