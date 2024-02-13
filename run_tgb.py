import os
from confs.conf_tgb_link_prediction import ctan_wiki, ctan_review, ctan_coin, ctan_comment


folder='./tgb_exp'

ncpu=1
ngpu=0.
cluster=True

# TGB-Wiki
wiki = (f'nohup python3 main_tgb_ctan.py --num-cpus-per-task {ncpu} --num-gpus-per-task {ngpu} ' 
       f'--memory_per_task 1500000000 {"--cluster" if cluster else ""} '
       f'--model {ctan_wiki} '
       '--lr_scheduler '
       '--lr_scheduler_patience 20 '
       '--epochs 200 '
       '--validate_every 1 '
       '--num_runs 5 '
       '--patience 50 '
       f'--data_name tgbl-wiki --data_dir {os.path.join(folder, "data")} --save_dir {folder} '
       f'>> {os.path.join(folder, "logs/results_wiki.log")} 2>&1 ')

# TGB-Review
review = (f'nohup python3 main_tgb_ctan.py --num-cpus-per-task {ncpu} --num-gpus-per-task {ngpu} ' 
       f'--memory_per_task 5000000000 {"--cluster" if cluster else ""} '
       f'--model {ctan_review} '
       '--lr_scheduler '
       '--lr_scheduler_patience 3 '
       '--epochs 50 '
       '--validate_every 3 '
       '--num_runs 3 '
       '--patience 10 '
       f'--data_name tgbl-review '
       f'--data_dir {os.path.join(folder, "data")} --save_dir {folder} '
       f'>> {os.path.join(folder, "logs/results_review.log")} 2>&1 ')

# TGB-Coin
coin = (f'nohup python3 main_tgb_ctan.py --num-cpus-per-task {ncpu} --num-gpus-per-task {ngpu} ' 
       f'--memory_per_task 5000000000 {"--cluster" if cluster else ""} '
       f'--model {ctan_coin} '
       '--lr_scheduler '
       '--lr_scheduler_patience 3 '
       '--epochs 50 '
       '--validate_every 3 '
       '--num_runs 3 '
       '--patience 10 '
       f'--data_name tgbl-coin '
       f'--data_dir {os.path.join(folder, "data")} --save_dir {folder} '
       f'>> {os.path.join(folder, "logs/results_coin.log")} 2>&1 ')

# TGB-Comment
comment = (f'nohup python3 main_tgb_ctan.py --num-cpus-per-task {ncpu} --num-gpus-per-task {ngpu} ' 
       f'--memory_per_task 5000000000 {"--cluster" if cluster else ""} '
       f'--model {ctan_comment} '
       '--lr_scheduler '
       '--lr_scheduler_patience 3 '
       '--epochs 50 '
       '--validate_every 3 '
       '--num_runs 3 '
       '--patience 20 '
       '--data_name tgbl-comment '
       f'--data_dir {os.path.join(folder, "data")} --save_dir {folder} '
       f'>> {os.path.join(folder, "logs/results_comment.log")} 2>&1 ')


with open('confs/__init__.py', 'w') as f:
    f.write('from .conf_tgb_link_prediction import *\n')
    f.flush()
    f.close()

for cmd in [wiki, review, coin, comment]:
    os.system(cmd)
