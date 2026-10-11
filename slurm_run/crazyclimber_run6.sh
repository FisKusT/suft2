#!/bin/bash
#SBATCH --job-name=crazyclimber6
#SBATCH --partition=shaham-L40
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=fiskust@biu.ac.il
#SBATCH --mail-type=END


cd /home/dsi/fiskustal/talfiskus/suft2/suft2 && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -u -m bbf.train --gin_files=bbf/configs/CrazyClimber.gin --base_dir=results/baseline_bbf_run6_CrazyClimber_100k --run_number=6 > "results/baseline_bbf_run6_CrazyClimber_100k.log" 2>&1

