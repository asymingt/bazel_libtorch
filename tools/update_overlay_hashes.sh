#!/usr/bin/env bash
# Recomputes the integrity hashes in <module>/<version>/source.json and keeps
# <module>/<version>/MODULE.bazel consistent with what the module's sources
# produce:
#
# - overlay/ (overlay modules): hashes every file into "overlay", and copies
#   overlay/MODULE.bazel to MODULE.bazel.
# - patches/ (patched modules): hashes every patch into "patches". If
#   patches/module_dot_bazel.patch creates MODULE.bazel (as in the BCR), it is
#   regenerated from MODULE.bazel.
#
# Usage: tools/update_overlay_hashes.sh modules/<name>/<version> [...]
#        tools/update_overlay_hashes.sh --check modules/<name>/<version> [...]
#
# With --check, nothing is written; the script exits non-zero if any file
# would change (for CI).
set -euo pipefail

check=0
if [[ "${1:-}" == "--check" ]]; then
  check=1
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--check] modules/<name>/<version> [...]" >&2
  exit 2
fi

exec python3 - "$check" "$@" <<'PY'
import base64, hashlib, json, os, sys

check = sys.argv[1] == "1"
status = 0


def integrity(path):
    with open(path, "rb") as f:
        return "sha256-" + base64.b64encode(hashlib.sha256(f.read()).digest()).decode()


def hash_tree(top, skip=()):
    hashes = {}
    for root, dirs, files in os.walk(top):
        dirs.sort()
        for name in sorted(files):
            if name in skip:
                continue
            path = os.path.join(root, name)
            hashes[os.path.relpath(path, top).replace(os.sep, "/")] = integrity(path)
    return dict(sorted(hashes.items()))


def module_patch(module_bazel):
    """A patch creating MODULE.bazel with the given content."""
    lines = module_bazel.splitlines()
    return "".join([
        "diff --git a/MODULE.bazel b/MODULE.bazel\n",
        "new file mode 100644\n",
        "--- /dev/null\n",
        "+++ b/MODULE.bazel\n",
        "@@ -0,0 +1,{} @@\n".format(len(lines)),
    ] + ["+" + line + "\n" for line in lines])


def sync(path, content):
    """Writes content to path, or reports drift with --check."""
    global status
    old = None
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
    if old == content:
        return
    if check:
        print(f"drift: {path} is out of date", file=sys.stderr)
        status = 1
        return
    with open(path, "w") as f:
        f.write(content)
    print(f"updated: {path}")


for dir in sys.argv[2:]:
    dir = dir.rstrip("/")
    source_path = os.path.join(dir, "source.json")
    overlay_dir = os.path.join(dir, "overlay")
    patches_dir = os.path.join(dir, "patches")
    if not os.path.isfile(source_path):
        sys.exit(f"error: {dir} has no source.json")

    with open(source_path) as f:
        source = json.load(f)
    updated = dict(source)

    if os.path.isdir(overlay_dir):
        with open(os.path.join(overlay_dir, "MODULE.bazel")) as f:
            sync(os.path.join(dir, "MODULE.bazel"), f.read())
        updated["overlay"] = hash_tree(overlay_dir, skip=("MODULE.bazel.lock",))

    if os.path.isdir(patches_dir):
        module_patch_path = os.path.join(patches_dir, "module_dot_bazel.patch")
        if os.path.exists(module_patch_path):
            with open(module_patch_path) as f:
                creates_module_bazel = "--- /dev/null\n+++ b/MODULE.bazel\n" in f.read()
            if creates_module_bazel:
                with open(os.path.join(dir, "MODULE.bazel")) as f:
                    sync(module_patch_path, module_patch(f.read()))
        updated["patches"] = hash_tree(patches_dir)

    sync(source_path, json.dumps(updated, indent=4) + "\n")

sys.exit(status)
PY
