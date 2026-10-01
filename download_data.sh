#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash download_data.sh [OUTPUT_DIR]"
  echo "Download, verify and extract the published dataset (default: repository data/)."
  exit 0
fi
if (( $# > 1 )) || [[ "${1:-}" == -* ]]; then
  echo "Usage: bash download_data.sh [OUTPUT_DIR]" >&2
  exit 2
fi
for tool in curl tar sha256sum cmp flock; do
  command -v "$tool" >/dev/null || { echo "Required command not found: $tool" >&2; exit 1; }
done

dataset_revision=2498b1dbfda63a8000d28aa1b5d504cf9ff6b21b
archive_name=far-prism-data.tar.gz
archive_sha256=eb1fbcd2ef2714218292f9faf98e6cf57ac7f792da0becf89b3ca89d65850f5d
archive_url="https://huggingface.co/datasets/Amazon-FAR/far-prism-data/resolve/$dataset_revision/$archive_name"
data_dir="${1:-$repo_root/data}"
mkdir -p -- "$data_dir"
data_dir="$(cd -- "$data_dir" && pwd)"
archive="$data_dir/$archive_name"
dataset_dir="$data_dir/far-prism-data"

# Serialize downloads and publish only fully verified files.
exec 9>"$data_dir/.prism-download.lock"
flock 9
work_dir="$(mktemp -d "$data_dir/.prism-download.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT

verify_archive() {
  printf '%s  %s\n' "$archive_sha256" "$1" | sha256sum --check --status
}

if [[ -e "$archive" || -L "$archive" ]]; then
  verify_archive "$archive" || { echo "Archive checksum mismatch: $archive" >&2; exit 1; }
else
  curl --fail --location --retry 3 --connect-timeout 30 --progress-bar \
    "$archive_url" --output "$work_dir/$archive_name"
  verify_archive "$work_dir/$archive_name" || { echo "Downloaded archive checksum mismatch." >&2; exit 1; }
  mv -- "$work_dir/$archive_name" "$archive"
fi

tar --no-same-owner --no-same-permissions -xzf "$archive" -C "$work_dir"
extracted="$work_dir/far-prism-data"
(cd -- "$extracted" && sha256sum --check SHA256SUMS --quiet)

if [[ -e "$dataset_dir" || -L "$dataset_dir" ]]; then
  # Preserve existing data and any separately added object assets.
  cmp --silent "$extracted/SHA256SUMS" "$dataset_dir/SHA256SUMS" || {
    echo "Existing dataset differs from this release; choose another OUTPUT_DIR." >&2
    exit 1
  }
  (cd -- "$dataset_dir" && sha256sum --check "$extracted/SHA256SUMS" --quiet)
else
  mv -T -- "$extracted" "$dataset_dir"
fi

echo "Dataset verified: $dataset_dir"
echo "Add the object meshes and prepare training shards as described in docs/data.md."
