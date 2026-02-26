#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar

dest_dir="${1:?dest_dir is required}"
wheel="${2:?wheel is required}"

mod_lib_dir=""
in_container=0
if [[ -f /.dockerenv ]]; then
  in_container=1
fi

if [[ -n "${MODULAR_LIB_DIR:-}" && -d "${MODULAR_LIB_DIR}" ]]; then
  mod_lib_dir="${MODULAR_LIB_DIR}"
fi

if [[ -z "${mod_lib_dir}" ]]; then
  patterns=(
    "/tmp/cibuildwheel/**/site-packages/modular/lib"
    "/tmp/**/site-packages/modular/lib"
    "/opt/python/*/lib/python*/site-packages/modular/lib"
    "/usr/local/lib/python*/site-packages/modular/lib"
  )

  if [[ "${in_container}" -eq 0 ]]; then
    patterns+=(
      "/project/.venv/lib/python*/site-packages/modular/lib"
      "./.venv/lib/python*/site-packages/modular/lib"
    )
  fi

  for pattern in "${patterns[@]}"; do
    for candidate in ${pattern}; do
      if [[ -d "${candidate}" ]]; then
        if [[ "${candidate}" == /project/.venv/* ]]; then
          continue
        fi
        mod_lib_dir="${candidate}"
        break
      fi
    done
    if [[ -n "${mod_lib_dir}" ]]; then
      break
    fi
  done
fi

if [[ -z "${mod_lib_dir}" ]]; then
  search_roots=(/tmp /opt/python /usr/local/lib /project)
  if [[ "${in_container}" -eq 0 ]]; then
    search_roots+=(.)
  fi

  found_lib="$(
    find "${search_roots[@]}" -type f -name "libKGENCompilerRTShared.so" 2>/dev/null \
      | rg -v "/project/.venv/" \
      | head -n1 || true
  )"
  if [[ -n "${found_lib}" ]]; then
    mod_lib_dir="$(dirname "${found_lib}")"
  fi
fi

if [[ -z "${mod_lib_dir}" ]]; then
  echo "ERROR: Could not locate modular runtime library directory." >&2
  exit 1
fi

echo "Using modular runtime libs from: ${mod_lib_dir}"

required_libs=(
  "libKGENCompilerRTShared.so"
  "libAsyncRTRuntimeGlobals.so"
  "libMSupportGlobals.so"
  "libNVPTX.so"
  "libAsyncRTMojoBindings.so"
)

for lib in "${required_libs[@]}"; do
  if [[ ! -f "${mod_lib_dir}/${lib}" ]]; then
    echo "ERROR: Missing required runtime library: ${mod_lib_dir}/${lib}" >&2
    exit 1
  fi
done

if ! command -v patchelf >/dev/null 2>&1; then
  echo "ERROR: patchelf is required for wheel repair but was not found in PATH." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

python -m wheel unpack "${wheel}" -d "${tmp_dir}"
unpacked_root="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"

if [[ -z "${unpacked_root}" || ! -d "${unpacked_root}/mogemma" ]]; then
  echo "ERROR: Could not locate unpacked mogemma wheel root." >&2
  exit 1
fi

wheel_pkg_dir="${unpacked_root}/mogemma"
wheel_lib_dir="${unpacked_root}/mogemma.libs"
mkdir -p "${wheel_lib_dir}"

for lib in "${required_libs[@]}"; do
  cp -av "${mod_lib_dir}/${lib}" "${wheel_lib_dir}/${lib}"
done

patchelf --set-rpath '$ORIGIN:$ORIGIN/../mogemma.libs' "${wheel_pkg_dir}/_core.so"

for lib in "${required_libs[@]}"; do
  patchelf --set-rpath '$ORIGIN' "${wheel_lib_dir}/${lib}"
done

python -m wheel pack "${unpacked_root}" -d "${dest_dir}"
