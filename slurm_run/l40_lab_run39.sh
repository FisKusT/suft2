#!/bin/bash
#SBATCH --job-name=suft_L39
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END


cd /home/dsi/fiskustal/talfiskus/dopamine && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 nohup python -u -m bbf.train --agent=BBF --gin_files=bbf/configs/BBF_search_config3.gin --base_dir=suft2/suft_1000_tf_1_0_huber_noisy_run9_Kangaroo_v5 --agent_seed=540160312  --no_seeding=False --run_number=79 --game_name="Kangaroo" > "suft2/suft_1000_tf_1_0_huber_noisy_run9_Kangaroo_v5.log" 2>&1 &

wait
