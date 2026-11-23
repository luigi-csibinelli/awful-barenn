#!/bin/bash
#SBATCH --job-name=barenn_cpu
#SBATCH --output=resultados/cpu_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --mem=2G
# Uso: sbatch scripts/franky_cpu.slurm <rotulo>   (ex.: v0.1)
# Pede o no inteiro (24 CPUs) para nenhum outro job interferir na medicao.
set -e
ROT=${1:-sem_rotulo}
export OMP_NUM_THREADS=1

echo "=== $ROT | $(date) | job $SLURM_JOB_ID ==="
lscpu | grep -E 'Model name|^CPU\(s\)|L1d|L2|L3'
echo "flags: $(grep -E '^CXXFLAGS' Makefile)"

make clean && make
scripts/validate.sh ./barenn
scripts/bench.sh ./barenn 10
scripts/profile_cpu.sh "$ROT"
