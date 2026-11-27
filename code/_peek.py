import json
import sys

for f in sys.argv[1:]:
    d = json.load(open(f, encoding="utf-8"))
    print(f.split("/")[-1], "beta=", d.get("beta"), "gamma=", d.get("gamma"),
          "vt=", d.get("verifier_top"), "h1=", round(d.get("hits1", -1), 4),
          "path=", bool(d.get("verifier_path")))
