#!/usr/bin/env bash
# Refresh the `integrity` field in modules/<name>/<version>/source.json by
# downloading the archive at its `url` and hashing it.
#
# Usage: tools/update_source_integrity.sh modules/qt/6.10.2 [more module dirs...]
set -euo pipefail

for dir in "$@"; do
  [[ -f "$dir/source.json" ]] || { echo "skip: $dir has no source.json" >&2; continue; }
  python3 - "$dir" <<'EOF'
import base64, hashlib, json, os, sys, urllib.request
d = sys.argv[1]
p = os.path.join(d, "source.json")
s = json.load(open(p))
url = s.get("url")
if not url:
    print(f"skip: {d} is not an archive-type module", file=sys.stderr)
    sys.exit(0)
with urllib.request.urlopen(url) as r:
    data = r.read()
integrity = "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()
old = s.get("integrity")
s["integrity"] = integrity
# keep key order stable: integrity, strip_prefix, url, then the rest (overlay, ...)
ordered = {k: s[k] for k in ("integrity", "strip_prefix", "url") if k in s}
ordered.update({k: v for k, v in s.items() if k not in ordered})
with open(p, "w") as fh:
    json.dump(ordered, fh, indent=4)
    fh.write("\n")
print(f"{d}: {len(data)} bytes from {url}\n  {old}\n  -> {integrity}")
EOF
done
