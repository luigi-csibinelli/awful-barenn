#!/bin/bash
#SBATCH --job-name=barenn_omp
#SBATCH --output=resultados/omp_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=2G
# Uso (no head-node):  make && sbatch scripts/franky_omp.slurm <rotulo>
# O binario e compilado no head-node (os nos nao tem os headers do compilador).
set -e
ROT=${1:-sem_rotulo}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=== $ROT | $(date) | job $SLURM_JOB_ID ==="
lscpu | grep -E 'Model name|^CPU\(s\)|Core|Socket|L1d|L2|L3'
[ -x ./barenn ] || { echo "ERRO: rode 'make' no head-node antes de submeter"; exit 1; }

echo "--- validacao: B=1 deve ser IDENTICO ao original (com 4 threads)"
OMP_NUM_THREADS=4 BARENN_BATCH=1 scripts/validate.sh ./barenn
echo "--- validacao: B=32 (mini-batch muda a ordem das contas -> acuracia)"
OMP_NUM_THREADS=4 BARENN_BATCH=32 scripts/validate.sh ./barenn acuracia

echo "--- escalabilidade (B=32)"
for T in 1 2 4 8 12 16; do
  OMP_NUM_THREADS=$T BARENN_BATCH=32 scripts/bench.sh ./barenn 5 | tail -2
done
echo "--- tamanho do lote (8 threads)"
for B in 8 16 32 64; do
  OMP_NUM_THREADS=8 BARENN_BATCH=$B ./barenn 2>/dev/null | grep -E 'Epoca 51|Tempo|Acuracia' | tr '\n' ' '; echo " <- B=$B"
done
echo "--- custo do modo lote com B=1 (1 thread)"
OMP_NUM_THREADS=1 BARENN_BATCH=1 scripts/bench.sh ./barenn 5 | tail -1
