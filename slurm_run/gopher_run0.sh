#!/bin/bash
#SBATCH --job-name=gopher0
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END


cd /home/dsi/fiskustal/talfiskus/suft2/suft2 && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files=bbf/configs/Gopher.gin --base_dir=results/baseline_bbf_run0_Gopher_100k --run_number=0 > "results/baseline_bbf_run0_Gopher_100k.log" 2>&1

