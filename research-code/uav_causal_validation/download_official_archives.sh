#!/usr/bin/env bash
set -euo pipefail

DATA_URL="https://zenodo.org/records/4379168/files/dataset.zip?download=1"
DATA_SIZE=72084530
DATA_MD5="d0dbf0bfb4891a3f34fcb2e381971087"
CHUNK_SIZE=1000000
OUT_DIR="${1:-/private/tmp/epfl_uav_official}"
PART_DIR="$OUT_DIR/dataset.parts"
DATA_ZIP="$OUT_DIR/dataset.zip"

mkdir -p "$PART_DIR"

start=0
while [ "$start" -lt "$DATA_SIZE" ]; do
  end=$((start + CHUNK_SIZE - 1))
  if [ "$end" -ge "$DATA_SIZE" ]; then
    end=$((DATA_SIZE - 1))
  fi
  expected=$((end - start + 1))
  part="$PART_DIR/${start}-${end}.part"
  header="$part.headers"
  if [ ! -f "$part" ] || [ "$(stat -f '%z' "$part")" -ne "$expected" ]; then
    ok=0
    for attempt in 1 2 3 4 5 6 7 8; do
      rm -f "$part.tmp" "$header"
      if curl --http1.1 --fail --silent --show-error --retry 3 --retry-all-errors \
          --retry-delay 2 --connect-timeout 20 -L \
          -D "$header" -H "Range: bytes=${start}-${end}" -o "$part.tmp" "$DATA_URL" \
          && [ -f "$part.tmp" ] \
          && grep -Eiq "^content-range: bytes ${start}-${end}/${DATA_SIZE}" "$header" \
          && [ "$(stat -f '%z' "$part.tmp")" -eq "$expected" ]; then
        ok=1
        break
      fi
      sleep $((attempt * 2))
    done
    [ "$ok" -eq 1 ]
    mv "$part.tmp" "$part"
  fi
  start=$((end + 1))
done

: > "$DATA_ZIP.tmp"
start=0
while [ "$start" -lt "$DATA_SIZE" ]; do
  end=$((start + CHUNK_SIZE - 1))
  if [ "$end" -ge "$DATA_SIZE" ]; then
    end=$((DATA_SIZE - 1))
  fi
  part="$PART_DIR/${start}-${end}.part"
  dd if="$part" bs=1048576 status=none >> "$DATA_ZIP.tmp"
  start=$((end + 1))
done

[ "$(stat -f '%z' "$DATA_ZIP.tmp")" -eq "$DATA_SIZE" ]
[ "$(md5 -q "$DATA_ZIP.tmp")" = "$DATA_MD5" ]
unzip -tq "$DATA_ZIP.tmp" >/dev/null
mv "$DATA_ZIP.tmp" "$DATA_ZIP"
printf '%s\n' "$DATA_ZIP"
