#!/usr/bin/env python
"""Emit LaTeX table fragments for the supplementary material from the raw
results files, so every printed number is machine-generated. Output:
reports/supplementary/tables/*.tex
"""
from __future__ import annotations

import json
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "supplementary" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def sci(x, digits=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def pct(x):
    return f"{100*x:.0f}\\%"


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def w(name, text):
    (OUT / name).write_text(text)
    print("wrote", name)


def jload(p):
    with open(p) as f:
        return json.load(f)


def runs(exp):
    return [jload(f) for f in sorted(glob.glob(str(ROOT / f"results/{exp}/*/result.json")))]


# ---------------------------------------------------------------- catalog
def make_catalog():
    cat = jload(ROOT / "data/verified_branches/branch_catalog.json")
    prof = {"(0,0)": "constant", "(1,0)": "bright sech", "(1,1)": "dark tanh kink",
            "(-1,-1)": "coth (singular)", "(0,-1)": "csch (singular)",
            "(1,-1)": "$2\\,\\mathrm{csch}(2\\xi)$ (singular)",
            "(-1,0)": "cosh (unbounded)", "(-1,1)": "$\\tfrac12\\sinh(2\\xi)$ (unbounded)",
            "(0,1)": "sinh (unbounded)"}
    rows = []
    for r in sorted(cat, key=lambda r: (r["group"], r["n"], r["m"])):
        key = f"({r['n']},{r['m']})"
        cons = r["constraints"]
        zeroed = [k for k, v in cons.items() if v == "0" and k not in ("lambda1",)]
        nonzero = {k: v for k, v in cons.items() if v != "0"}
        cons_s = ", ".join(f"${esc(k)}={esc(v)}$".replace("lambda", "\\lambda_")
                           .replace("beta", "\\beta_").replace("alpha2", "\\alpha_2")
                           .replace("gamma", "\\gamma").replace("Gamma", "\\Gamma")
                           .replace("ell", "\\ell").replace("asq", "a^2")
                           .replace("**", "^").replace("*", " ")
                           for k, v in nonzero.items() if k != "lambda1")
        lam1 = cons.get("lambda1", "free")
        rows.append(
            f"$({r['n']},{r['m']})$ & {prof.get(key,'--')} & {r['tier']} & {r['group']} & "
            f"{len(zeroed)} & {sci(float(r['residual_max']))} & "
            f"{sci(float(r['residual_rms_rel']))} \\\\")
    body = "\n".join(rows)
    w("catalog.tex", f"""\\begin{{tabular}}{{llllrrr}}
\\toprule
$(n,m)$ & profile & tier & group & \\#coeffs forced 0 & $\\max|F|$ & rel.\\ RMS \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- numerical
def make_numerical():
    s = jload(ROOT / "results/numerical/summary.json")
    rows = []
    for bid, entry in s.items():
        name = {"n0_m0_f0": "constant $(0,0)$", "n1_m0_f0": "sech $(1,0)$",
                "n1_m1_f0": "kink $(1,1)$"}.get(bid, esc(bid))
        for r in entry["primary"]["runs"]:
            rows.append(f"{name} & fd\\_mol & {r['nt']} & {sci(r['rtol'])} & "
                        f"{sci(r['final_l2re'])} & {sci(r['final_max_err'])} & "
                        f"{r['runtime_s']:.2f} \\\\")
            name = ""
        if "cross_check_fourier" in entry:
            r = entry["cross_check_fourier"]
            rows.append(f" & ETDRK4 & {r['nt']} & {r['nz_steps']} steps & "
                        f"{sci(r['final_l2re'])} & {sci(r['final_max_err'])} & "
                        f"{r['runtime_s']:.2f} \\\\")
        oo = ", ".join(f"{o:.1f}" for o in entry["primary"]["observed_orders"])
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\quad\\footnotesize observed spatial orders: {oo}}} \\\\[2pt]")
    body = "\n".join(rows)
    w("numerical.tex", f"""\\begin{{tabular}}{{llrrrrr}}
\\toprule
branch & scheme & $n_t$ & tol / steps & final L2RE & max err & runtime (s) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- forward
def make_forward():
    done = [r for r in runs("forward_pilot") if r.get("state") == "done"]
    cells = {}
    for r in done:
        cells.setdefault((r["branch"], r["method"]), []).append(r)
    order_b = ["(0,0)", "(1,0)", "(1,1)"]
    name_b = {"(0,0)": "constant $(0,0)$", "(1,0)": "bright sech $(1,0)$",
              "(1,1)": "dark kink $(1,1)$"}
    name_m = {"vanilla": "fixed weights", "adaptive_weights": "adaptive weights",
              "adaptive_sampling": "adaptive sampling"}
    rows = []
    for b in order_b:
        first = True
        for m in ["vanilla", "adaptive_weights", "adaptive_sampling"]:
            rs = cells.get((b, m), [])
            v = [x["l2re_complex"] for x in rs]
            res = [x["pde_residual_rms_independent"] for x in rs]
            sr = np.mean([x["success"] for x in rs])
            rows.append(f"{name_b[b] if first else ''} & {name_m[m]} & {len(v)} & "
                        f"{sci(np.median(v))} & {sci(np.mean(v))} $\\pm$ {sci(np.std(v))} & "
                        f"{sci(min(v))} -- {sci(max(v))} & {pct(sr)} & {sci(np.median(res))} \\\\")
            first = False
        rows.append("\\addlinespace")
    body = "\n".join(rows)
    w("forward_matrix.tex", f"""\\begin{{tabular}}{{llrrrrrr}}
\\toprule
branch & method & $n$ & median & mean $\\pm$ std & range & success & resid.\\ RMS \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")

    # hi-fi
    hf = [r for r in runs("forward_sech_hifi") if r.get("state") == "done"]
    by = {}
    for r in hf:
        by.setdefault(r["method"], []).append(r)
    rows = []
    for m in ["vanilla", "adaptive_weights", "adaptive_sampling"]:
        rs = by.get(m, [])
        v = [x["l2re_complex"] for x in rs]
        rows.append(f"{name_m[m]} & {len(v)} & {sci(np.median(v))} & "
                    f"{sci(min(v))} -- {sci(max(v))} & "
                    f"{pct(np.mean([x['success'] for x in rs]))} \\\\")
    allv = [x["l2re_complex"] for x in hf]
    rows.append("\\midrule")
    rows.append(f"all methods & {len(allv)} & {sci(np.median(allv))} & "
                f"{sci(min(allv))} -- {sci(max(allv))} & "
                f"{pct(np.mean([x['success'] for x in hf]))} \\\\")
    w("hifi.tex", "\\begin{tabular}{lrrrr}\n\\toprule\n"
      "method & $n$ & median L2RE & range & success \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

    # stress
    st = [r for r in runs("forward_stress") if r.get("state") == "done"]
    by = {}
    for r in st:
        by.setdefault(r["branch"], []).append(r)
    prof = {"(-1,-1)": "coth kink (singular)", "(0,-1)": "csch pulse (singular)",
            "(1,-1)": "$2\\,\\mathrm{csch}(2\\xi)$ (singular)",
            "(-1,0)": "cosh (unbounded, $\\beta_3,\\beta_4$ active)",
            "(-1,1)": "$\\tfrac12\\sinh(2\\xi)$ (unbounded, travelling)",
            "(0,1)": "sinh (unbounded, travelling)"}
    rows = []
    for b in sorted(by):
        v = [x["l2re_complex"] for x in by[b]]
        rows.append(f"$({b.strip('()')})$ & {prof.get(b,'--')} & {len(v)} & "
                    f"{sci(np.median(v))} & {sci(min(v))} -- {sci(max(v))} & "
                    f"{pct(np.mean([x['success'] for x in by[b]]))} \\\\")
    w("stress.tex", "\\begin{tabular}{llrrrr}\n\\toprule\n"
      "$(n,m)$ & profile & $n$ & median L2RE & range & success \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


# ---------------------------------------------------------------- inverse
def make_inverse():
    rows_all = jload(ROOT / "results/inverse_pilot/param_rows.json")
    pname = {"c": "$c=g_0T_2^2$", "ell": "$\\ell=(g_0-\\alpha)/2$",
             "alpha2": "$\\alpha_2$", "gamma": "$\\gamma$ (true 0)",
             "Gamma": "$\\Gamma$ (true 0)", "beta2": "$\\beta_2$ (true 0)",
             "beta3": "$\\beta_3$ (true 0)", "beta4": "$\\beta_4$ (true 0)"}

    def med(stage, param, noise=None, density=None):
        v = []
        for r in rows_all:
            if r["stage"] != stage or r["param"] != param:
                continue
            if noise is not None and r["noise"] != noise:
                continue
            if density is not None and r["density"] != density:
                continue
            x = r["rel_error"] if (r["rel_error"] is not None
                                   and np.isfinite(r["rel_error"])) else r["abs_error"]
            v.append(x)
        return (np.median(v), len(v)) if v else (None, 0)

    # noise table at d=5%
    noises = [0.0, 0.01, 0.05]
    rows = []
    for stage, params in [("gainloss", ["c", "ell"]),
                          ("nonlinear", ["gamma", "Gamma", "alpha2"]),
                          ("dispersion", ["beta2", "beta3", "beta4"])]:
        first = True
        for p in params:
            cells = [sci(med(stage, p, noise=nv, density=0.05)[0]) for nv in noises]
            n = med(stage, p, noise=0.0, density=0.05)[1]
            rows.append(f"{stage if first else ''} & {pname[p]} & "
                        + " & ".join(cells) + f" & {n} \\\\")
            first = False
        rows.append("\\addlinespace")
    w("inverse_noise.tex", "\\begin{tabular}{llrrrr}\n\\toprule\n"
      "stage & parameter & 0\\% noise & 1\\% noise & 5\\% noise & runs \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

    # density table (gainloss)
    rows = []
    for p in ["c", "ell"]:
        cells = [sci(med("gainloss", p, density=d)[0]) for d in [0.01, 0.05, 0.10]]
        rows.append(f"{pname[p]} & " + " & ".join(cells) + " \\\\")
    w("inverse_density.tex", "\\begin{tabular}{lrrr}\n\\toprule\n"
      "parameter & 1\\% density & 5\\% density & 10\\% density \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

    # joint table
    rows = []
    for p in ["c", "ell", "alpha2", "Gamma"]:
        mj, nj = med("joint", p)
        ms, _ = med({"c": "gainloss", "ell": "gainloss",
                     "alpha2": "nonlinear", "Gamma": "nonlinear"}[p], p, density=0.05)
        rows.append(f"{pname[p]} & {sci(ms)} & {sci(mj)} & {nj} \\\\")
    w("inverse_joint.tex", "\\begin{tabular}{lrrr}\n\\toprule\n"
      "parameter & staged (median) & joint 4-param (median) & joint runs \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

    # identifiability
    ident = jload(ROOT / "results/inverse_pilot/identifiability.json")
    rows = []
    for bid, d in ident.items():
        name = {"n1_m1_f0": "kink $(1,1)$", "n1_m0_f0": "sech $(1,0)$"}.get(bid, esc(bid))
        nd = ", ".join(f"{x:.3f}" for x in d["g0T2alpha_null_direction"])
        rows.append(f"{name} & {d['reduced_rank_1e-8']}/8 & "
                    f"{sci(d['reduced_condition_number'])} & "
                    f"{d['g0T2alpha_rank_1e-8']}/3 & $({nd})$ \\\\")
    w("identifiability.tex", "\\begin{tabular}{lrrrl}\n\\toprule\n"
      "branch & reduced rank & cond.\\ number & $(g_0,T_2,\\alpha)$ rank & null direction \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


# ---------------------------------------------------------------- ablations
def make_ablations():
    abl = jload(ROOT / "results/ablations/ablation_summary.json")
    label = {"saturation::Gamma0": "saturation off ($\\Gamma=0$)",
             "saturation::GammaPos": "saturation on ($\\Gamma=1$)",
             "fourth_order::beta4_0": "$\\beta_4=0$",
             "fourth_order::beta4_pos": "$\\beta_4=0.4$"}
    rows = []
    for k, st in abl["summary"].items():
        rows.append(f"{label.get(k, esc(k))} & {sci(st['mean'])} $\\pm$ {sci(st['std'])} "
                    f"& {st['n_finite']} \\\\")
    w("ablations.tex", "\\begin{tabular}{lrr}\n\\toprule\n"
      "arm (constant branch, one closure-consistent change) & mean L2RE $\\pm$ std & $n$ \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


# ---------------------------------------------------------------- generalization
def make_generalization():
    gen = jload(ROOT / "results/generalization_pilot/all_results.json")
    branches = ["n0_m0_f0", "n1_m0_f0", "n1_m1_f0"]
    bl = {"n0_m0_f0": "constant $(0,0)$", "n1_m0_f0": "sech $(1,0)$",
          "n1_m1_f0": "kink $(1,1)$"}
    modes = [("zero_shot", "zero-shot"), ("fine_tune_0.01", "fine-tune 1\\%"),
             ("fine_tune_0.05", "fine-tune 5\\%"),
             ("scratch_same_budget", "scratch, same budget")]
    rows = []
    for b in branches:
        cells = []
        for key, _ in modes:
            v = []
            for res in gen:
                e = res.get("lobo", {}).get(b, {}).get(key, {})
                if isinstance(e, dict):
                    v += list(e.values())
            cells.append(sci(np.mean(v)) if v else "--")
        rows.append(f"{bl[b]} & " + " & ".join(cells) + " \\\\")
    # interp/extrap row
    for key, lab in [("interp", "interpolation (all branches)"),
                     ("extrap", "extrapolation (all branches)")]:
        v = []
        for res in gen:
            for b in branches:
                x = res.get(key, {}).get(b)
                if isinstance(x, (int, float)):
                    v.append(x)
        rows.append(f"\\multicolumn{{5}}{{l}}{{{lab}: mean L2RE {sci(np.mean(v))}}} \\\\")
    head = " & ".join(lab for _, lab in modes)
    w("generalization.tex", "\\begin{tabular}{lrrrr}\n\\toprule\n"
      f"held-out branch & {head} \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    make_catalog()
    make_numerical()
    make_forward()
    make_inverse()
    make_ablations()
    make_generalization()
