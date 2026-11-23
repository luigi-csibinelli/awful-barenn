#!/usr/bin/env bash
# Uso: scripts/bench.sh <executavel> [repeticoes] [arquivo_treino arquivo_teste]
# Faz 1 execucao de aquecimento (descartada) e depois N execucoes medidas,
# reportando a media, o minimo e o maximo do "Tempo de treino".
set -euo pipefail
BIN=${1:?informe o executavel}
N=${2:-10}
shift $(( $# >= 2 ? 2 : $# ))
ARGS=("$@")

echo "host: $(hostname)"
echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
echo "compilador: $(g++ --version | head -1)"
echo "binario: $BIN | repeticoes: $N | OMP_NUM_THREADS: ${OMP_NUM_THREADS:-n/a} | BARENN_BATCH: ${BARENN_BATCH:-n/a}"

"$BIN" ${ARGS[@]+"${ARGS[@]}"} > /dev/null 2>&1   # aquecimento

for i in $(seq "$N"); do
  "$BIN" ${ARGS[@]+"${ARGS[@]}"} 2>/dev/null | sed -n 's/^Tempo de treino \([0-9.]*\) s$/\1/p'
done | awk '
  NR == 1 { min = $1; max = $1 }
  { s += $1; if ($1 < min) min = $1; if ($1 > max) max = $1 }
  END { printf "tempo de treino (s): media %.4f | min %.4f | max %.4f | %d execucoes\n", s/NR, min, max, NR }'
