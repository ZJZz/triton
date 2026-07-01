#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="$root/docs/ai/THINKING.md"
target_file="$root/AGENTS.md"
begin_marker="<!-- BEGIN: docs/ai/THINKING.md -->"
end_marker="<!-- END: docs/ai/THINKING.md -->"

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

awk \
  -v source_file="$source_file" \
  -v begin_marker="$begin_marker" \
  -v end_marker="$end_marker" '
BEGIN {
  while ((getline line < source_file) > 0) {
    source = source line ORS
  }
  close(source_file)
}
$0 == begin_marker {
  begin_count++
  print
  printf "%s", source
  skipping = 1
  next
}
$0 == end_marker {
  end_count++
  skipping = 0
  print
  next
}
!skipping {
  print
}
END {
  if (begin_count != 1 || end_count != 1 || skipping) {
    exit 1
  }
}
' "$target_file" > "$tmp_file"

mv "$tmp_file" "$target_file"
trap - EXIT
