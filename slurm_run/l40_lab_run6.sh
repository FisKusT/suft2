#!/bin/bash
#SBATCH --job-name=suft_L6
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END


cd /home/dsi/fiskustal/talfiskus/dopamine && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 nohup python -u -m bbf.train --agent=BBF --gin_files=bbf/configs/BBF_search_config1.gin --base_dir=suft2/suft_optimization_500_tf_1_0_huber_run6_Kangaroo_v5 --agent_seed=1642206906 --no_seeding=False --run_number=76 --game_name="Kangaroo" > "suft2/suft_optimization_500_tf_1_0_huber_run6_Kangaroo_v5.log" 2>&1 &

wait