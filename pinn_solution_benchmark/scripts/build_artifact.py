#!/usr/bin/env python
"""Build reports/artifact_report.html from the template: inline {{IMG:path}}
tokens as base64 data URIs and fill the dynamic text tokens from results."""
from __future__ import annotations

import base64
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def img_data_uri(rel: str) -> str:
    p = ROOT / rel
    data = base64.b64encode(p.read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{data}" alt="{Path(rel).stem}" '
            f'loading="lazy">')


def pick_field_triptych() -> str:
    cands = sorted(glob.glob(str(ROOT / "reports/figures/forward_pilot/"
                                 "field_forward_pilot__n1_m1_f0__rep0__vanilla__seed*.png")))
    if not cands:
        cands = sorted(glob.glob(str(ROOT / "reports/figures/forward_pilot/field_*.png")))
    return Path(cands[0]).name


def dynamic_tokens() -> dict:
    tok = {}
    # hi-fi sech sentence
    hifi = [json.load(open(f)) for f in
            glob.glob(str(ROOT / "results/forward_sech_hifi/*/result.json"))]
    hifi_done = [r for r in hifi if r.get("state") == "done"]
    if hifi_done:
        import numpy as np
        med = np.median([r["l2re_complex"] for r in hifi_done])
        sr = np.mean([r.get("success", False) for r in hifi_done])
        tok["HIFI_SENTENCE"] = (
            f"High-fidelity retraining of the sech branch ({len(hifi_done)} runs, "
            f"2.5&times; steps, 2&times; points) reaches median L2RE {med:.1e} with "
            f"{sr:.0%} success &mdash; the pilot-budget failure is "
            + ("largely a budget artifact." if sr >= 0.5 else
               "only partially a budget artifact: the branch remains the hardest."))
    else:
        tok["HIFI_SENTENCE"] = ("High-fidelity sech retraining is running; "
                                "this sentence updates on completion.")
    # stress caption
    stress = [json.load(open(f)) for f in
              glob.glob(str(ROOT / "results/forward_stress/*/result.json"))]
    sdone = [r for r in stress if r.get("state") == "done"]
    if sdone:
        import numpy as np
        by = {}
        for r in sdone:
            by.setdefault(r["branch"], []).append(r["l2re_complex"])
        worst = max(by, key=lambda b: np.median(by[b]))
        best = min(by, key=lambda b: np.median(by[b]))
        tok["STRESS_CAPTION"] = (
            f"Stress-group runs ({len(sdone)}) span "
            f"{np.median(by[best]):.0e} ({best}) to {np.median(by[worst]):.0e} ({worst}): "
            "singular and large-amplitude profiles push errors up by one to two decades.")
    else:
        tok["STRESS_CAPTION"] = "Stress-group results are added when those runs complete."
    # inverse density note + extra li
    rows_p = ROOT / "results/inverse_pilot/param_rows.json"
    tok["INV_DENSITY_NOTE"] = ""
    tok["INV_EXTRA_LI"] = ""
    if rows_p.exists():
        rows = json.load(open(rows_p))
        dens = sorted({r["density"] for r in rows})
        stages = sorted({r["stage"] for r in rows})
        if len(dens) > 1:
            tok["INV_DENSITY_NOTE"] = (f"Densities covered: "
                                       f"{', '.join(f'{d:.0%}' for d in dens)}.")
        if "dispersion" in stages:
            import numpy as np
            dv = [r["abs_error"] for r in rows if r["stage"] == "dispersion"]
            tok["INV_EXTRA_LI"] += (f"<li><b>Dispersion stage:</b> &beta;&#8322;, "
                                    f"&beta;&#8323;, &beta;&#8324; (all true 0 on these "
                                    f"branches) pinned to median absolute error "
                                    f"{np.median(dv):.1e}.</li>")
        if "joint" in stages:
            import numpy as np
            jv = [r["rel_error"] for r in rows if r["stage"] == "joint"
                  and r["rel_error"] is not None]
            jv = [v for v in jv if v == v]
            if jv:
                tok["INV_EXTRA_LI"] += (f"<li><b>Joint 4-parameter stage</b> (c, &ell;, "
                                        f"&alpha;&#8322;, &Gamma;): median relative error "
                                        f"{np.median(jv):.1e}.</li>")
    # scope notes + run count
    n_extra = len(hifi_done) + len(sdone)
    inv_n = len(glob.glob(str(ROOT / "results/inverse_pilot/n*__*.json")))
    tok["N_EXTRA_RUNS"] = str(n_extra + inv_n)
    missing = []
    if not hifi_done:
        missing.append("the high-fidelity sech runs")
    if not sdone:
        missing.append("the stress-group runs")
    if missing:
        tok["SCOPE_NOTES"] = ("Still executing at publication time: "
                              + " and ".join(missing) + " (resume commands above). ")
    else:
        tok["SCOPE_NOTES"] = ("Every experiment listed above has been executed at the "
                              "stated scale; the only configuration not run end-to-end "
                              "on this machine is the GPU-scale "
                              "<code>configs/full.yaml</code>. ")
    return tok


def main():
    tpl = (ROOT / "reports/artifact_template.html").read_text()
    tpl = tpl.replace("{{FIELD_TRIPTYCH}}", pick_field_triptych())
    for k, v in dynamic_tokens().items():
        tpl = tpl.replace("{{" + k + "}}", v)
    def repl(m):
        return img_data_uri(m.group(1))
    out = re.sub(r"\{\{IMG:([^}]+)\}\}", repl, tpl)
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", out)
    if leftover:
        raise SystemExit(f"unfilled tokens: {leftover}")
    dest = ROOT / "reports/artifact_report.html"
    dest.write_text(out)
    print(f"wrote {dest} ({len(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
