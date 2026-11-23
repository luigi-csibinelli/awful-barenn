#!/bin/bash
#SBATCH --job-name=barenn_cpu
#SBATCH --output=resultados/cpu_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=2G
# Uso (no head-node):  make && sbatch scripts/franky_cpu.slurm <rotulo>
# Profiling (no head-node, o valgrind so existe la):  scripts/profile_cpu.sh <rotulo>
# Os nos de computacao nao tem os headers do compilador: o binario e
# compilado no head-node ANTES de submeter; o job so executa e mede.
# Pede o no inteiro (16 CPUs) para nenhum outro job interferir na medicao.
set -e
ROT=${1:-sem_rotulo}
export OMP_NUM_THREADS=1

echo "=== $ROT | $(date) | job $SLURM_JOB_ID | no $(hostname) ==="
lscpu | grep -E 'Model name|^CPU\(s\)|L1d|L2|L3'
echo "flags: $(grep -E '^CXXFLAGS' Makefile)"
[ -x ./barenn ] || { echo "ERRO: rode 'make' no head-node antes de submeter"; exit 1; }

scripts/validate.sh ./barenn
scripts/bench.sh ./barenn 10
