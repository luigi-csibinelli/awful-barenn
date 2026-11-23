#!/bin/bash
#SBATCH --job-name=barenn_gpu
#SBATCH --output=resultados/gpu_%j.out
#SBATCH --partition=merry_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=4G
# Uso (no head-node):  module load cuda/12.8.1 && make barenn_cuda && sbatch scripts/franky_gpu.slurm <rotulo> [kernels_ncu]
# O binario e compilado no head-node (os nos nao tem os headers do compilador).
# kernels_ncu = quantos lancamentos de kernel o Nsight Compute analisa.
set -e
ROT=${1:-sem_rotulo}
NCU_K=${2:-1}
module load cuda/12.8.1

echo "=== $ROT | $(date) | job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
nvcc --version | tail -1
lscpu | grep -E 'Model name'

[ -x ./barenn_cuda ] || { echo "ERRO: rode 'make barenn_cuda' no head-node antes de submeter"; exit 1; }
scripts/validate.sh ./barenn_cuda
scripts/bench.sh ./barenn_cuda 10

mkdir -p profiling
echo "--- Nsight Systems"
nsys profile --trace=cuda --sample=none --stats=true --force-overwrite=true \
  -o "profiling/nsys_$ROT" ./barenn_cuda > "profiling/nsys_$ROT.txt" 2>&1
grep -A 12 -E 'cuda_gpu_kern_sum|cuda_gpu_mem_time_sum|cuda_api_sum' "profiling/nsys_$ROT.txt" || true

echo "--- Nsight Compute ($NCU_K kernel(s))"
if ncu --set full -c "$NCU_K" -f -o "profiling/ncu_$ROT" ./barenn_cuda > /dev/null 2>&1; then
  ncu --import "profiling/ncu_$ROT.ncu-rep" --page details > "profiling/ncu_$ROT.txt"
  grep -E 'Duration|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Shared Memory Configuration Size|Block Size|Waves Per SM|Memory Throughput|Compute \(SM\) Throughput|Bank' "profiling/ncu_$ROT.txt" || true
else
  echo "ncu falhou (provavel falta de permissao para os contadores da GPU)"
fi
