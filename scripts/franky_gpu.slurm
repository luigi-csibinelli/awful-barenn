#!/bin/bash
#SBATCH --job-name=barenn_gpu
#SBATCH --output=resultados/gpu_%j.out
#SBATCH --partition=merry_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G
# A fila merry_gpu aceita no maximo 5 minutos por job, por isso cada etapa e
# um job separado. O binario e compilado no head-node antes de submeter:
#   module load cuda/12.8.1 && make barenn_cuda
# Uso:
#   sbatch scripts/franky_gpu.slurm <rotulo> bench [repeticoes]  validacao + tempo (padrao 10)
#   sbatch scripts/franky_gpu.slurm <rotulo> nsys                Nsight Systems (ate 60 s de coleta)
#   sbatch scripts/franky_gpu.slurm <rotulo> ncu [kernels]       Nsight Compute (padrao 1 kernel)
set -e
ROT=${1:-sem_rotulo}
MODO=${2:-bench}
N=${3:-}
module load cuda/12.8.1

echo "=== $ROT | $MODO | $(date) | job $SLURM_JOB_ID | no $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
[ -x ./barenn_cuda ] || { echo "ERRO: rode 'make barenn_cuda' no head-node antes de submeter"; exit 1; }
mkdir -p profiling

case "$MODO" in
  bench)
    scripts/validate.sh ./barenn_cuda
    scripts/bench.sh ./barenn_cuda "${N:-10}"
    ;;
  nsys)
    nsys profile --trace=cuda --sample=none --stats=true --force-overwrite=true \
      --duration=60 -o "profiling/nsys_$ROT" ./barenn_cuda > "profiling/nsys_$ROT.txt" 2>&1 || true
    grep -A 12 -E 'cuda_gpu_kern_sum|cuda_gpu_mem_time_sum|cuda_api_sum' "profiling/nsys_$ROT.txt" || true
    ;;
  ncu)
    if ncu --set full -c "${N:-1}" --kill yes -f -o "profiling/ncu_$ROT" ./barenn_cuda \
         > "profiling/ncu_${ROT}_log.txt" 2>&1; then
      ncu --import "profiling/ncu_$ROT.ncu-rep" --page details > "profiling/ncu_$ROT.txt"
      grep -E 'Duration|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Block Size|Grid Size|Waves Per SM|Memory Throughput|Compute \(SM\) Throughput|Bank' "profiling/ncu_$ROT.txt" || true
    else
      echo "ncu falhou; ultimas linhas do log:"
      tail -5 "profiling/ncu_${ROT}_log.txt"
    fi
    ;;
  *)
    echo "modo invalido: $MODO (use bench, nsys ou ncu)"; exit 2 ;;
esac
