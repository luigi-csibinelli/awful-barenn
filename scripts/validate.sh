#!/usr/bin/env bash
# Uso: scripts/validate.sh <executavel> [exato|acuracia]
#   exato    (padrao): saida identica a referencia (todas as perdas, o
#                      checksum dos pesos e a avaliacao), exceto o tempo.
#   acuracia          : acuracia no teste no maximo 0,5 p.p. abaixo da
#                      referencia (para versoes que mudam a ordem das contas).
set -euo pipefail
BIN=${1:?informe o executavel}
MODO=${2:-exato}
REF=ref/saida_referencia.txt
OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

"$BIN" 2>/dev/null | grep -v '^Tempo de treino' > "$OUT"

case "$MODO" in
  exato)
    if diff -u "$REF" "$OUT"; then
      echo "OK: saida identica a referencia"
    else
      echo "FALHOU: saida diferente da referencia"; exit 1
    fi ;;
  acuracia)
    ref=$(sed -n 's/^Acuracia \([0-9.]*\)%.*/\1/p' "$REF")
    obt=$(sed -n 's/^Acuracia \([0-9.]*\)%.*/\1/p' "$OUT")
    if awk -v r="$ref" -v o="$obt" 'BEGIN { exit !(o != "" && o + 0 >= r - 0.5) }'; then
      echo "OK: acuracia $obt% (referencia $ref%)"
    else
      echo "FALHOU: acuracia $obt% (referencia $ref%)"; exit 1
    fi ;;
  *)
    echo "modo invalido: $MODO (use exato ou acuracia)"; exit 2 ;;
esac
