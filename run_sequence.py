import os
import ray
from datasets import SYNTHETIC_SEQUENCE

@ray.remote(num_cpus=1)
def run_cmd(cmd):
    return os.system(cmd)


folder='./temporal_path_graph'
num_runs=1
num_final_runs=5
epochs=20
patience=5
cluster=True
log=True
cpus_per_task=1
gpus_per_task=0.

ray.init()

with open('confs/__init__.py', 'w') as f:
    f.write('from .conf_sequence import *\n')
    f.flush()
    f.close()

ray_ids = []
for data in SYNTHETIC_SEQUENCE:
    for model in ['TGAT', 'DyRep', 'JODIE', 'TGN', 'CTAN']:

        cmd = (f"python3 -u main.py --data_dir {folder}/DATA --data_name {data} --save_dir {folder}/RESULTS_sequence "
               f"--model {model} --num_runs {num_runs} --num_final_runs {num_final_runs} --epochs {epochs} --patience {patience} "
               f"--num_cpus_per_task {cpus_per_task} --num_gpus_per_task {gpus_per_task} --batch 128"
               f"{'--cluster' if cluster else ''} --verbose {'--log' if log else ''} "
               f"> {folder}/out_{model}_{data} 2> {folder}/err_{model}_{data}")

        print('Running:', cmd)
        ray_ids.append(run_cmd.remote(cmd))
        
        while len(ray_ids) > 4:
            done_id, ray_ids = ray.wait(ray_ids)

while ray_ids:
    done_id, ray_ids = ray.wait(ray_ids)