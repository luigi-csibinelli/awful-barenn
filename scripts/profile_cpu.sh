#!/usr/bin/env bash
# Uso: scripts/profile_cpu.sh <rotulo>   (ex.: v0.1)
# callgrind: custo (instrucoes) por funcao e por linha.
# cachegrind: acessos e falhas de cache (L1 e ultimo nivel).
# Saidas em profiling/.
# Sem pipefail: o "| head" fecha o pipe cedo e isso nao e erro.
set -eu
ROT=${1:?informe um rotulo, ex.: v0.1}
mkdir -p profiling

valgrind --tool=callgrind --callgrind-out-file="profiling/callgrind.$ROT.out" ./barenn > /dev/null
{
  echo "### custo proprio (exclusivo) por funcao"
  callgrind_annotate "profiling/callgrind.$ROT.out" | head -45
  echo
  echo "### custo inclusivo por funcao"
  callgrind_annotate --inclusive=yes "profiling/callgrind.$ROT.out" | head -45
} > "profiling/callgrind.$ROT.txt"
echo "callgrind: profiling/callgrind.$ROT.txt"

valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file="profiling/cachegrind.$ROT.out" \
  ./barenn > /dev/null 2> "profiling/cachegrind_resumo.$ROT.txt"
cg_annotate "profiling/cachegrind.$ROT.out" > "profiling/cachegrind.$ROT.txt" 2>/dev/null || true
echo "cachegrind: profiling/cachegrind_resumo.$ROT.txt (D1/LL miss rate) e profiling/cachegrind.$ROT.txt"
