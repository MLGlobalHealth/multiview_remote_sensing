#!/bin/bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem-per-cpu=64G   
#SBATCH --time=24:00:00     
#SBATCH --job-name=random_bands
tual folder path!
cd /data/coml-satellites/exet6440/multiview_remote_sensing_test/


# ==========================================
# 3. RUN THE PIPELINE
# ==========================================
echo "Starting rigorous fine-tuning and evaluation pipeline..."

# Run the master pipeline script hiding inside the dino folder.
# Make sure to change the imagery_path to your actual imagery folder!
# Run the master pipeline script hiding inside the dino folder.
python modelling/dino/run_random_experiment.py \
    --imagery_path /data/coml-satellites/satellite_imagery \
    --num_random_tests 3
[exet6440@arc-login01 multiview_remote_sensing_test-main]$ 
echo "Job finished!"