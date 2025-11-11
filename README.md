# Initializing

Set the data input path and the results path as ...   
```bash
source ./set_vars.bash /sdf/data/slac/s2ai/waveforms/twoamps_at_home /sdf/data/slac/s2ai/waveforms/results
python3 src/collect_waves.py
```

if using slurm for batch processing with a number of directories...  

```bash
sbatch slurmscript.bash /sdf/data/slac/s2ai/waveforms/Waveforms_2019-09-06/tens_0_1000_2*
```
I think this globs the arguments and then processes them sequentially.  Not sure how long this will take.  


