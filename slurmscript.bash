#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mail-user=coffee@slac.stanford.edu
#SBATCH -N1
#SBATCH -n1
#SBATCH -o ../output.out
#SBATCH --job-name=waves
#SBATCH --account=lcls
#SBATCH --partition=roma

echo "processing $# directories"
source $HOME/s3ai/bin/activate
for dirname in "$@"; do
	echo $dirname
	source ./set_vars.bash $dirname /sdf/data/slac/s2ai/waveforms/results
	python3 src/collect_waves.py
done
