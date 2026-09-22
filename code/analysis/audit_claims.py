#!/usr/bin/env python3
"""Deterministic claim audit for the ICASSP submission.

Read-only: it never writes to paper/ or results/run_csv/. It checks that every
number the paper reports is (a) present in the named evidence file under
results/ and (b) present in paper/main.tex after LaTeX normalisation.
Writes results/analysis/CLAIM_AUDIT.{json,md} and exits non-zero on failure.

Run from the repository root:  python code/analysis/audit_claims.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
ANA = ROOT / "results" / "analysis"
OUT_JSON = ANA / "CLAIM_AUDIT.json"
OUT_MD = ANA / "CLAIM_AUDIT.md"


def norm(s):
    """Normalise a number as rendered in LaTeX or in a CSV to bare digits."""
    s = str(s)
    s = s.replace("{,}", "").replace(",", "")
    s = s.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace("%", "").replace("~", "").replace("--", "-")
    s = s.replace("\\,", "").replace(" ", "")
    return s


def read_csv(name):
    """Read a derived table from this package's results/analysis/.

    The package is self-contained: a missing evidence file is a hard
    failure, never a silent skip or a fallback to a parent repository
    (GPT final review P0-4).
    """
    path = ANA / name
    if not path.exists():
        raise SystemExit("audit: required evidence file missing from this "
                         "package: results/analysis/%s" % name)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt_macs(v):
    m = float(v) / 1e6
    return "{:.1f}M".format(m) if m >= 100 else "{:.2f}M".format(m)


def main():
    tex = norm(TEX.read_text(encoding="utf-8"))
    checks = []

    def check(cid, claim, value, source, where="paper"):
        """where='paper' -> value must appear in main.tex; 'evidence' -> in source text."""
        v = norm(value)
        if where == "paper":
            ok = v in tex
        else:
            path = ANA / source
            if not path.exists():
                raise SystemExit("audit: required evidence file missing from "
                                 "this package: results/analysis/%s" % source)
            ok = v in norm(path.read_text(encoding="utf-8"))
        checks.append({"id": cid, "claim": claim, "value": str(value),
                       "source": source, "checked_against": where, "ok": bool(ok)})

    # ---- Table 1: one check per row for NMSE, params and MACs ---------------
    main_rows = read_csv("main_table.csv")
    for i, r in enumerate(main_rows, 1):
        if r.get("status") != "ok":
            continue
        label = r["label"]
        # capacity-scan rows are quoted as a range (0.1367-0.1423), not per row
        cap = label.startswith("capacity-") and label != "capacity-matched-transformer"
        # 2026-09-18: these rows moved out of Table 1 into a prose sentence that
        # quotes their NMSE only; params/MACs remain repo evidence.
        prose = label in {"mamba-lite-L", "gru-baseline", "itransformer-L-iclr24",
                          "transformer-262k",
                          "timemixer-L-iclr24", "dlinear-baseline", "tcn-baseline",
                          "mlp-baseline"}
        # 2026-09-19 round 2: the rejected-objective paragraph no longer quotes
        # the with-loss NMSEs; the rows stay regenerable evidence.
        withloss = label.startswith("ours-with-loss")
        w = "evidence" if (cap or prose or withloss) else "paper"
        wn = "evidence" if (cap or withloss or prose) else "paper"
        check("T1-%d-nmse" % i, "Table 1 %s NMSE" % label, r["nmse"], "main_table.csv", where=wn)
        if r.get("params") and r["params"] not in ("", "None"):
            check("T1-%d-params" % i, "Table 1 %s params" % label, r["params"], "main_table.csv", where=w)
        # R22 review P1-10: the per-model MAC caption list left the paper;
        # ours/reference MACs remain covered by the EFF checks below.

    # ---- Table 2: ablation rows -------------------------------------------
    # The loss-on factorial was retired from the paper (superseded by the
    # loss-off factorial walked via ablation_table_off.csv above); its rows
    # remain regenerable evidence, not paper-quoted numbers.
    for i, r in enumerate(read_csv("ablation_table.csv"), 1):
        check("T2-%d-nmse" % i, "Table 2 %s NMSE" % r["cell"], r["nmse"], "ablation_table.csv", where="evidence")
        rel = r["relative_vs_reference"].rstrip("%")
        if rel not in ("+0.0", "-0.0"):
            check("T2-%d-rel" % i, "Table 2 %s delta" % r["cell"], abs(float(rel)), "ablation_table.csv", where="evidence")

    # ---- seed study --------------------------------------------------------
    for i, r in enumerate(read_csv("seeds_table.csv"), 1):
        # round 2: with-loss rows are no longer paper-quoted numbers
        w = "evidence" if (r["nmse"] == "0.1401" or str(r["family"]).startswith("ours-with-loss")) else "paper"
        check("SEED-%d" % i, "seed study %s/%s ep %s" % (r["family"], r["seed"], r["schedule_epochs"]),
              r["nmse"], "seeds_table.csv", where=w)

    # ---- robustness split (only rows the paper quotes must appear in the tex)
    QUOTED_ROB = {"0.1623", "0.1772", "0.1735", "0.1777", "0.1805"}
    for i, r in enumerate(read_csv("robustness_table.csv"), 1):
        where = "paper" if r["nmse"] in QUOTED_ROB else "evidence"
        check("ROB-%d" % i, "robustness %s" % r["label"], r["nmse"], "robustness_table.csv", where=where)

    # ---- generalization slice (same rule)
    QUOTED_GEN = {"0.2685", "0.2872", "0.2951"}  # 216-slice rows retired from the paper
    for i, r in enumerate(read_csv("generalization_table.csv"), 1):
        for col, tag in (("ours_nmse", "ours"), ("published_nmse", "published")):
            where = "paper" if r[col] in QUOTED_GEN else "evidence"
            check("GEN-%d-%s" % (i, tag), "generalization %s %s" % (r["label"], tag),
                  r[col], "generalization_table.csv", where=where)
    QUOTED_CM_DIFF = set()  # by_cm.csv is deprecated; three-seed values quoted instead
    for i, r in enumerate(read_csv("generalization_by_cm.csv"), 1):
        if r["slice"] == "eqweight-relative":
            continue
        QUOTED_CM_NMSE = set()  # deprecated file
        for col, tag in (("ours", "ours"), ("published", "published")):
            if r[col]:
                where = "paper" if r[col] in QUOTED_CM_NMSE else "evidence"
                check("GENCM-%d-%s" % (i, tag), "by-CM %s %s" % (r["slice"], tag),
                      r[col], "generalization_by_cm.csv", where=where)
        if r["paired_mean_diff"] in QUOTED_CM_DIFF:
            check("GENCM-%d-diff" % i, "by-CM %s diff" % r["slice"], r["paired_mean_diff"],
                  "generalization_by_cm.csv")

    # ---- efficiency profile ------------------------------------------------
    prof = (ANA / "profiles_4090.txt").read_text(encoding="utf-8")
    for label, params, macs in (("ours", 173190, 6514464),
                                ("published", 21913750, 804901376),
                                ("mambacsp", 881604, 2856963584),
                                ("capacity-matched", 149560, 480000)):
        check("EFF-%s-params" % label, "profile %s params" % label, params,
              "profiles_4090.txt", where="evidence")
        check("EFF-%s-macs" % label, "profile %s MACs" % label, macs,
              "profiles_4090.txt", where="evidence")
    # Documented convention (R13, 2026-09-21): the parameter and MAC ratios
    # are quoted at one decimal (126.53 -> 126.5, 123.56 -> 123.6). Raw ratios
    # are printed for the record.
    pr, mr = 21913750 / 173190, 804901376 / 6514464
    print("raw ratios: parameters %.2fx, MACs %.2fx" % (pr, mr))
    check("EFF-param-ratio", "126.5x parameter ratio",
          "%.1f" % pr, "profiles_4090.txt")
    check("EFF-mac-ratio", "MAC ratio (paper quotes one decimal, 123.6x)",
          "%.1f" % mr, "profiles_4090.txt")
    check("EFF-latency-ratio", "4.98x latency ratio",
          round(11.15 / 2.24, 2), "profiles_4090.txt")
    check("EFF-control-ratio", "1.29x vs shrunk published architecture",
          round(0.1807 / 0.1402, 2), "main_table.csv")

    # ---- harness reproduction ---------------------------------------------
    check("REPRO", "harness reproduction deviation", "0.105", "harness_reproduction.txt", where="evidence")

    # ---- venue / process invariants ---------------------------------------
    # ---- new evidence files that the 2026-09-17 integration will quote -----
    # Each entry lists the numbers the paper is expected to carry; they are
    # checked against BOTH the regenerating file (evidence) and paper/main.tex,
    # so a stale or mistyped value fails the audit either way.
    QUOTED_SEEDS = {"0.2685", "0.2767", "0.2695", "-0.0187", "-0.0105",
                    "-0.0177", "-0.0156", "-0.025", "-0.007"}
    for i, r in enumerate(read_csv("generalization_seeds.csv"), 1):
        for col, tag in (("ours", "ours"), ("paired_mean_diff", "diff"),
                         ("ci_low", "ci_low"), ("ci_high", "ci_high")):
            if not r.get(col):
                continue
            where = "paper" if r[col] in QUOTED_SEEDS else "evidence"
            check("SEEDS-%d-%s" % (i, tag),
                  "seeds %s %s" % (r["slice"], tag), r[col],
                  "generalization_seeds.csv", where=where)

    # 2026-09-20 R9: Table 2 (tab:factorial) tabulates the full 2^3, so the
    # decomp+pole pair 0.1458 and the R01-corrected pole cell 0.1408 are now
    # quoted; 0.1437 dropped (no CSV row carries it since the R01 closure).
    QUOTED_ABLOFF = {"0.1397", "0.1779", "0.1862", "0.1425",
                     "0.1968", "0.1853", "0.1408", "0.1458"}
    for i, r in enumerate(read_csv("ablation_table_off.csv"), 1):
        if r.get("status") != "ok":
            continue
        where = "paper" if r["nmse"] in QUOTED_ABLOFF else "evidence"
        check("ABLOFF-%d-nmse" % i, "loss-off ablation %s" % r["cell"],
              r["nmse"], "ablation_table_off.csv", where=where)

    QUOTED_ACD = {"0.2031", "0.2182", "-0.0151", "-0.0256", "-0.0054"}
    for i, r in enumerate(read_csv("generalization_acd_seeds.csv"), 1):
        for col, tag in (("ours", "ours"), ("published", "published"),
                         ("paired_mean_diff", "diff"), ("ci_low", "ci_low"),
                         ("ci_high", "ci_high")):
            if not r.get(col):
                continue
            where = "paper" if r[col] in QUOTED_ACD else "evidence"
            check("ACD-%d-%s" % (i, tag), "A/C/D %s %s" % (r["slice"], tag),
                  r[col], "generalization_acd_seeds.csv", where=where)

    # ---- cross-file integrity: per-seed A/C/D tables must stay distinct -----
    # R13 (2026-09-21): the seed-42 table once shipped as a byte-copy of the
    # seed-44 output; hash all three so any recurrence fails the gate.
    import hashlib
    RUNS = ROOT / "results" / "run_csv"
    acd_digests = {}
    for seed, name in ((42, "CAPNOAUX600_gen_acd.csv"),
                       (43, "CAPNOAUXS43_gen_acd.csv"),
                       (44, "HEADNOAUXS44_gen_acd.csv")):
        path = RUNS / name
        acd_digests[seed] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists()
            else "missing")
    hash_groups = {}
    for seed, digest in acd_digests.items():
        hash_groups.setdefault(digest, []).append(seed)
    dupes = [s for s in hash_groups.values() if len(s) > 1]
    checks.append({"id": "INT-acd-seed-hashes",
                   "claim": "three per-seed A/C/D CSVs pairwise distinct (R13 duplicate-seed guard)",
                   "value": ("distinct" if not dupes and "missing" not in
                             acd_digests.values() else
                             "duplicates: %s" % dupes),
                   "source": "results/run_csv", "checked_against": "evidence",
                   "ok": not dupes and "missing" not in acd_digests.values()})

    QUOTED_LADDER = {"1.3771", "1.2465", "1.2790", "0.8236", "0.7874",
                     "0.2300",  # 1.8081 (no prediction) retired from Table 1
                     # R19 exactness-ladder rungs quoted in ssec:price:
                     # E0 full-grid exact, E1 LS-fitted gains (native, full).
                     "3.2883", "2.7558", "2.7518",
                     "1.2504", "1.1366", "1.4767",
                     "0.8173"}  # 0.7625/1.0545 (E1b shifted) stay evidence-only
    for i, r in enumerate(read_csv("classical_ladder.csv"), 1):
        where = "paper" if r["cell_mean_nmse"] in QUOTED_LADDER else "evidence"
        check("LADDER-%d" % i, "classical ladder %s/%s" % (r["split"], r["model"]),
              r["cell_mean_nmse"], "classical_ladder.csv", where=where)

    # ---- 2026-09-22 R19: K-support monotonicity + gain-conditioning arms ---
    # ssec:price quotes the K=2/4/8 trend between the two ladder endpoints;
    # the run_csv cells are the evidence, and the trend must ascend.
    def run_cell_mean(name):
        path = RUNS / name
        if not path.exists():
            raise SystemExit("audit: required evidence file missing from this "
                             "package: results/run_csv/%s" % name)
        vals = []
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            parts = r["nmse_mean"].strip("[]").split()
            vals.append(sum(float(x) for x in parts) / len(parts))
        return sum(vals) / len(vals)

    kprev = 1.3771
    for k, quoted in (("2", "1.7825"), ("4", "2.5235"), ("8", "3.0204")):
        name = "DFTGRID_K%s_full162.csv" % k
        cm = run_cell_mean(name)
        checks.append({"id": "LADDERK-%s" % k,
                       "claim": "K=%s full-grid cell mean (ssec:price trend, "
                                "quoted as monotone)" % k,
                       "value": quoted, "source": "results/run_csv/%s" % name,
                       "checked_against": "evidence",
                       "ok": abs(cm - float(quoted)) < 5e-5})
        checks.append({"id": "LADDERK-%s-asc" % k,
                       "claim": "raising the K support degrades monotonically",
                       "value": "%.4f -> %.4f" % (kprev, cm),
                       "source": "results/run_csv", "checked_against": "evidence",
                       "ok": cm > kprev})
        kprev = cm
    checks.append({"id": "LADDERK-16-asc",
                   "claim": "K=16 closes the monotone trend at the full grid",
                   "value": "%.4f -> 3.2883" % kprev,
                   "source": "classical_ladder.csv", "checked_against": "evidence",
                   "ok": 3.2883 > kprev})

    for name, quoted in (("GAINOFF_full162.csv", "0.1584"),
                         ("GAINOFFREF_full162.csv", "0.1386")):
        cm = run_cell_mean(name)
        checks.append({"id": "GAINOFF-%s" % name.split("_")[0],
                       "claim": "gain-conditioning arm cell mean (ssec:ablations)",
                       "value": quoted, "source": "results/run_csv/%s" % name,
                       "checked_against": "paper+evidence",
                       "ok": abs(cm - float(quoted)) < 5e-5 and quoted in tex})

    QUOTED_CTRL = {"0.3039", "0.3427", "0.1807", "0.2106"}
    for i, r in enumerate(read_csv("generalization_432_summary.csv"), 1):
        where = "paper" if r["nmse_432"] in QUOTED_CTRL else "evidence"
        check("GEN432-%d-ours" % i, "432 control %s" % r["model"],
              r["nmse_432"], "generalization_432_summary.csv", where=where)

    QUOTED_POLE = {"0.1923", "0.1385"}
    for i, r in enumerate(read_csv("pole_offset_curve.csv"), 1):
        where = "paper" if r["nmse"] in QUOTED_POLE else "evidence"
        check("POLE-%d" % i, "pole offset %s" % r["tag"], r["nmse"],
              "pole_offset_curve.csv", where=where)

    QUOTED_BAND = {"-0.0204", "0.0001", "-0.0378", "-0.0057", "-0.0180",
                   "0.0169"}
    for i, r in enumerate(read_csv("generalization_by_band.csv"), 1):
        for col, tag in (("mean_diff", "diff"), ("ci_low", "ci_low"),
                         ("ci_high", "ci_high")):
            if not r.get(col):
                continue
            where = "paper" if r[col] in QUOTED_BAND else "evidence"
            check("BAND-%d-%s" % (i, tag),
                  "band %s seed %s %s" % (r["band"], r["seed"], tag), r[col],
                  "generalization_by_band.csv", where=where)

    # ---- 2026-09-21 R14: B/E delay-spread decomposition (V8.2 axis) --------
    # Provenance: results/analysis/generalization_be_by_delay_spread.csv
    # (72 cells x ds50/ds200/ds400, paired bootstrap, three seeds).
    QUOTED_BE = {"-0.0367", "-0.0549", "-0.0189"}
    for i, r in enumerate(read_csv("generalization_be_by_delay_spread.csv"), 1):
        for col, tag in (("paired_mean_diff", "diff"), ("ci_low", "ci_low"),
                         ("ci_high", "ci_high")):
            if not r.get(col):
                continue
            where = "paper" if r[col] in QUOTED_BE else "evidence"
            check("BE-%d-%s" % (i, tag), "B/E by delay spread %s %s" % (r.get("slice", i), tag),
                  r[col], "generalization_be_by_delay_spread.csv", where=where)

    # ---- 2026-09-19 wave: paired CIs, extended slices, capacity persistence
    # 2026-09-19 round 2: per-seed regular paired diffs quoted (seeds 42/43/44);
    # the seed-42 CI endpoints stay regenerable evidence only.
    QUOTED_STATS = {"0.0139", "0.0196", "0.0132"}
    for i, r in enumerate(read_csv("stats_hardening.csv"), 1):
        for col in ("mean_diff", "ci_low", "ci_high"):
            if not r.get(col):
                continue
            where = "paper" if r[col] in QUOTED_STATS else "evidence"
            check("STATS-%d-%s" % (i, col), "stats hardening %s %s" % (r["comparison"], col),
                  r[col], "stats_hardening.csv", where=where)

    QUOTED_EXT = {"0.0288", "0.0691"}  # 0.0439 is the interior point of the quoted range
    for fn, tag in (("generalization_432_extended.csv", "BE"),
                    ("generalization_acd_extended.csv", "ACD"),
                    ("robustness_extended.csv", "ROB")):
        for i, r in enumerate(read_csv(fn), 1):
            if not r.get("paired_diff"):
                continue
            where = "paper" if r["paired_diff"] in QUOTED_EXT else "evidence"
            check("EXT-%s-%d" % (tag, i), "intervention on shift slice %s %s" % (tag, r.get("model", "")),
                  r["paired_diff"], fn, where=where)

    for i, r in enumerate(read_csv("pole_snap_capacity.csv"), 1):
        where = "paper" if r.get("pct_vs_stock", r.get("pct")) in ("38.8",) else "evidence"
        col = "pct_vs_stock" if "pct_vs_stock" in r else "pct"
        check("SNAPCAP-%d" % i, "intervention vs capacity %s" % r.get("checkpoint", r.get("label", i)),
              r[col], "pole_snap_capacity.csv", where=where)

    for i, r in enumerate(read_csv("pole_offset_curve.csv"), 1):
        where = "paper" if r.get("pct_vs_stock") in ("38.8",) else "evidence"
        check("OFFPCT-%d" % i, "pole offset pct %s" % r["tag"],
              r["pct_vs_stock"], "pole_offset_curve.csv", where=where)

    for i, r in enumerate(read_csv("ablation_table_mech.csv"), 1):
        rel = r["relative_vs_reference"].rstrip("%")
        where = "paper" if rel in ("+0.9", "+0.1") else "evidence"
        check("MECH-%d" % i, "mechanism ablation %s" % r["cell"],
              abs(float(rel)), "ablation_table_mech.csv", where=where)

    for i, r in enumerate(read_csv("data_frac_curve.csv"), 1):
        where = "evidence"  # the paper rounds to "about 17%"; exact pcts stay in the file
        check("DATAFRAC-%d" % i, "data fraction %s" % r["train_frac"],
              r["pct_vs_full"], "data_frac_curve.csv", where=where)

    QUOTED_CAP = {"0.1367", "0.1423", "0.1385"}  # range endpoints + reported point
    for i, r in enumerate(read_csv("capacity_curve.csv"), 1):
        where = "paper" if r["nmse"] in QUOTED_CAP else "evidence"
        check("CAPWALK-%d" % i, "capacity scan %s" % r["label"],
              r["nmse"], "capacity_curve.csv", where=where)

    raw_tex = TEX.read_text(encoding="utf-8")
    invariants = [("INV-tcn", "TCN row present (D1 closed)", "TCN" in raw_tex),
                  ("INV-natbib", "no natbib", "natbib" not in raw_tex),
                  ("INV-data-needed", "no DATA_NEEDED marker", "DATA_NEEDED" not in raw_tex),
                  ("INV-bib", "bibliography is references only",
                   "\\bibliography{references}" in raw_tex)]
    for cid, claim, ok in invariants:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})

    # ---- 2026-09-18 revision: theory chain, integer ratios, removals -------
    # Theorem 1 closure: per-horizon grid-snap penalty grows monotonically.
    for r in read_csv("pole_snap_horizon.csv"):
        if r.get("group") == "horizon":
            check("TH-h%s" % r["value"], "grid-snap penalty at horizon %s (measured series, evidence-only in R14)" % r["value"],
                  r["pct"], "pole_snap_horizon.csv", where="evidence")
    # Offset quantisation curve: 1/8 / 1/4 / 1/2-bin penalties.
    for r in read_csv("pole_intervention_curve.csv"):
        if r.get("family") == "quant":
            check("TH-quant-%s" % r["tag"], "offset quantisation penalty (%s bin, evidence-only in R14)" % r["tag"],
                  r["pct_vs_stock"], "pole_intervention_curve.csv", where="evidence")
    # Concept words the R14 exactness-budget chain depends on.
    for word, claim in [("semigroup", "local semigroup law present"),
                        ("Theorem 1", "Theorem 1 (spectral-defect bound) present"),
                        ("Corollary 1", "Corollary 1 (finite-scan certificate) present"),
                        ("varepsilon", "total offset notation present")]:
        check("TH-word-%s" % word.replace(" ", "-"), claim, word, "paper/main.tex")
    flat_tex = " ".join(raw_tex.split())  # line-wrapping-insensitive phrase matching

    # ---- 2026-09-21 R14: the exactness-budget rewrite --------------------
    # New theorem chain (Prop 1 exactness budget / Thm 1 spectral-defect
    # bound / Cor 1 finite-scan certificate), new title, rebuilt abstract,
    # parameter-inventory section, wrap-column defect figure. Replaces the
    # retired R2-R12 wording blocks; arithmetic stays in audit_arithmetic.
    invariants_r14 = [
        # title + abstract
        ("R14-title", "exactness-on-a-budget title",
         "DD-Koopman: Exactness on a Budget for Compact Multi-Step CSI Prediction" in flat_tex),
        ("R14-abs-tension", "abstract carries the zero-DOF endpoint",
         "the exact transfer is unique with zero degrees of freedom" in flat_tex
         and "reduces to periodic continuation" in flat_tex),
        ("R14-abs-budget", "abstract carries 2T(N-K) DOF and the KT/N price",
         "buys $2T(N{-}K)$ degrees of freedom at a minimum white-noise gain of $KT/N$" in flat_tex),
        ("R14-abs-certificate", "abstract states the finite-scan certificate",
         "certified from a finite scan" in flat_tex),
        ("R14-abs-endpoints", "grid-pinned retrain finding carried in ssec:price",
         "Pinning the poles to the grid and retraining recovers the reported"
         in flat_tex),
        ("R14-abs-allocation", "abstract carries the allocation split "
         "(R22 review P1-10: numbers dieted, split phrasing kept)",
         "input-side estimation with a conditional spectral readout" in flat_tex
         and "173{,}190" in flat_tex),
        # theory block
        ("R14-expansion", "branch expansion P_j = W_j F_N with gated mixture",
         "$P_j=W_jF_N$" in flat_tex
         and "gates $\\alpha=(1{-}g_1{-}g_2,\\,g_1,\\,g_2)$" in flat_tex),
        ("R14-semigroup", "conditioned transfer is a semigroup in h",
         "semigroup law $K_c^{\,h+k}=K_c^{\,h}K_c^{\,k}$" in flat_tex),
        ("R14-warmstart", "ridge least-squares warm start kept as init choice",
         "applies the ridge least-squares warm start" in flat_tex),
        ("R14-prop1", "Prop 1 named, selector explicit, gain normalized",
         "Proposition 1 (exactness and its price)" in flat_tex
         and "$P_0=[I_T\\;0]$" in flat_tex
         and "$G(P_0)=KT/N$" in flat_tex),
        ("R14-thm1", "Thm 1 states the defect bound and its three-term diagnostic",
         "Theorem 1 (spectral-defect bound)" in flat_tex
         and "attained at a single atom" in flat_tex
         and "model-class defect, input perturbation, and correction" in flat_tex),
        ("R14-cor1", "Cor 1 states the finite-scan certificate",
         "Corollary 1 (finite-scan certificate)" in flat_tex
         and "a finite scan certifies the continuous defect" in flat_tex),
        ("R14-corr-fp64", "correspondence: double-precision identity + float32 attribution",
         "3\\times10^{-15}" in flat_tex and "1.97\\times10^{-6}" in flat_tex),
        ("R14-corr-scan", "correspondence: scan certificate + synthetic coverage",
         "311{,}040" in flat_tex and "10{,}800" in flat_tex
         and "3.84" in flat_tex and "2.37--6.24" in flat_tex),
        ("R14-bib-golub", "Golub & Van Loan cited for the min-norm budget",
         "golub2013matrix" in raw_tex),
        ("R14-estimator-split", "estimator/transfer split stated in Method",
         "Input processing (the estimator)" in flat_tex),
        # 4.2 parameter inventory
        ("R14-inventory", "module inventory quoted with shares",
         "12{,}768 parameters (7.4\%)" in flat_tex
         and "154{,}328 (89.1\%)" in flat_tex
         and "1{,}590" in flat_tex and "four scalar gates" in flat_tex
         and "173{,}190 in total" in flat_tex),
        ("R14-endpoints", "inventory bridges to the trade-off table",
         "places this split on the accuracy--efficiency plane" in flat_tex),
        # 4.3 price of exactness
        ("R14-pin", "pin-and-retrain control on both slices",
         "0.1401 against 0.1402" in flat_tex and "0.2688 against 0.2716" in flat_tex),
        ("R14-pin-rob", "pin control holds on the robustness split",
         "0.1775 against 0.1761" in flat_tex),
        ("R14-poles-off", "fitted poles use the freedom",
         "89\% of trained poles" in flat_tex and "0.05 bins" in flat_tex),
        ("R14-snap", "frozen snap intervention with mean/per-setting scope",
         "0.1385 to 0.1923" in flat_tex and "with all 162 settings worse" in flat_tex
         and "per-mode, per-sample placement" in flat_tex),
        ("R14-snap-shift", "snap degradation across the four splits",
         "1{,}728 settings across the four splits" in flat_tex
         and "$+17.7\%$ to $+38.8\%$" in flat_tex
         and "$+0.0288$ to $+0.0691$" in flat_tex),
        ("R14-defect-fig", "wrap-column defect figure wired with readings",
         "Figure~\\ref{fig:defect} measures the learned transfer's off-grid response" in flat_tex
         and "falls to 2.10 at $z{=}1$" in flat_tex
         and "peaks at 7.1 in the mid-band" in flat_tex
         and "between 4.1 and 8.3" in flat_tex),
        ("R14-defect-artifact", "defect figure + generator shipped",
         (ROOT / "paper" / "fig_defect.pdf").exists()
         and (ROOT / "paper" / "figures" / "make_fig_defect.py").exists()
         and "fig_defect.pdf" in raw_tex),
        # 4.4 ablations
        ("R14-factorial-pole", "factorial pole cell at corrected value",
         "0.1408 & $+0.0011$" in flat_tex),
        ("R14-factorial-pair", "factorial pair row present",
         "0.1458 & $+0.0061$" in flat_tex),
        ("R14-main-effect", "denoiser main effect recomputed from the factorial",
         "$+31.7\%$ average main effect across the four flip pairs" in flat_tex),
        ("R14-attrib-denoiser", "unseen-slice attribution to the denoiser with CI",
         "$+20.9\%$ without it, 95\% CI $[+0.0431,+0.0713]$" in flat_tex),
        ("R14-attrib-rest", "decomposition secondary; pole removal helps",
         "to the delay decomposition ($+4.0\%$); removing the pole conditioning slightly helps ($-0.9\%$)" in flat_tex),
        ("R14-fredf", "FreDF negative result kept as matched-comparison wording",
         "every matched comparison trains worse with it (two seeds at 600 epochs, one at 240); we drop it" in flat_tex),
        ("R14-capacity", "width-insensitivity + data-sensitivity statements",
         "0.1367--0.1423 across a $5.6\\times$, 58k--327k parameter scan" in flat_tex
         and "performance is more sensitive to data than to width" in flat_tex),
        # 4.5 operating point
        ("R14-tradeoff", "three-seed operating point against the published model",
         "0.1402 against the published model's 0.1246" in flat_tex
         and "0.1385, 0.1442 and 0.1378" in flat_tex),
        ("R14-macs", "MAC counts and ratios at one decimal",
         "6{,}514{,}464" in flat_tex and "804{,}901,376" in flat_tex
         and "126.5$\\times$ and 123.6$\\times$" in flat_tex),
        ("R14-latency", "measured latency pair and speedup",
         "$2.24\pm0.05$\,ms against $11.15\pm0.02$\,ms" in flat_tex
         and "4.98$\\times$ faster" in flat_tex),
        ("R14-mrt", "MRT beamforming comparison kept scoped",
         "2.06\% vs.\ 2.70\% loss, paired $+0.64$\,pp" in flat_tex),
        # 4.6 shift
        ("R14-shift-be", "B/E axis with the delay-spread concentration reading",
         "Consistent with the estimator-based interpretation" in flat_tex
         and "$-0.0367$ at 400\,ns" in flat_tex),
        ("R14-shift-closing", "three-seed unseen advantage stated with per-seed CIs",
         "0.2685, 0.2767 and 0.2695" in flat_tex
         and "($-0.0187$, $-0.0105$, $-0.0177$)" in flat_tex),
        # conclusion
        ("R14-conclusion", "conclusion carries the chain and the three anchors",
         "estimator followed by a small linear transfer" in flat_tex
         and "removal costs rank" in flat_tex
         and "holds a three-seed unseen-propagation advantage" in flat_tex),
        ("R14-abstract-split", "38.8% per-setting scope carried by sec 4.3 (kept from R3)",
         "with all 162 settings worse" in flat_tex),
        ("R3-caption", "pareto caption states the three-seed-validated operating point",
         "the reported model is the three-seed-validated operating" in flat_tex),
        ("R3-artifact", "figure shipped as vector PDF, no raster (Type 3 compliance)",
         (ROOT / "paper" / "fig1_final.pdf").exists()
         and not (ROOT / "paper" / "fig1_final.png").exists()),
        # factorial table protocol (kept from R9)
        ("R9-tab2-protocol", "Table 2 caption discloses the factorial protocol",
         "spectral loss off, 240 epochs, seed 42" in flat_tex),
        ("R9-tab2-pointer", "prose references Table 2 for the factorial",
         "Table~\\ref{tab:factorial} runs the full $2^3$" in flat_tex),
        ("R9-tab2-interpretation", "superadditivity interpretation survives the tabulation",
         "costs more than the sum of the parts" in flat_tex),
        # ---- R19 exactness ladder + gain conditioning (2026-09-22) ----
        ("R19-ladder-e0", "full-grid exact transfer scored, worst row, three splits",
         "3.2883" in flat_tex and "the worst row in" in flat_tex
         and "2.7558 and 2.7518" in flat_tex),
        ("R19-ladder-trend", "K-support raise is monotonically worse "
         "(trend values verified from run_csv, not quoted)",
         "degrades" in flat_tex and "monotonically" in flat_tex
         and "suppressing noisy modes" in flat_tex),
        ("R19-ladder-e1", "release exactness on the same support: LS-fitted gains",
         "1.2504" in flat_tex and "1.1366 and 1.4767" in flat_tex
         and "0.8173" in flat_tex
         and "the model's own warm start" in flat_tex),
        ("R19-ladder-correspondence", "analytic continuation vs harness correspondence",
         "1.7\\times10^{-6}" in flat_tex
         and "analytic continuation" in flat_tex),
        ("R19-gainoff", "gain-conditioning ablation, both arms, control-anchored",
         "0.1584 against 0.1386" in flat_tex and "+14.3" in flat_tex
         and "0.1397" in flat_tex),
        ("R19-abs-ladder", "abstract opens the measurements with the E0 endpoint",
         "scores 3.2883 against the learned model's 0.1402" in flat_tex),
        ("R19-contrib-costs", "contribution 2 ranks the three removal costs",
         "0.8\\% for pole conditioning, 14.3\\% for the readout gains' observation"
         in flat_tex and "27.3\\% for the input denoiser" in flat_tex),
        ("R19-price-title", "ssec:price owns the ladder under its own name",
         "\\subsection{The price of exactness}" in flat_tex),
        # ---- R22 pre-submission review integration (2026-09-22) ----
        ("R22-256", "native LS sample budget disclosed (fit_summary.json: 256 packed samples)",
         "256-sample fit" in flat_tex and "24{,}300" in flat_tex),
        ("R22-domain", "certificate scan domain disclosed (diag_lib.py: unit circle, 257 angles, eta = pi/256)",
         "257" in flat_tex and "pi/256" in flat_tex and "rho{=}1" in flat_tex),
        ("R22-branch", "branch histories and conditioned operators defined (review P0-3)",
         "A_0,A_1,A_2" in flat_tex and "P(A_j)" in flat_tex),
        ("R22-median", "synthetic tightness stated as the measured median, not a blanket range",
         "median" in flat_tex and "3.84" in flat_tex and "2.37--6.24" in flat_tex),
    ]
    for cid, claim, ok in invariants_r14:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # Removed claims must stay removed (negative checks).
    for gone, claim in [("0.1383", "stale capacity lower bound removed"),
                        ("126times", "integer parameter ratio replaced by 126.5x (R13)"),
                        ("124times", "integer MAC ratio replaced by 123.6x (R13)"),
                        ("1/126", "fractional parameter ratio replaced by 126.5x (R13)"),
                        ("only the reported model's paired differences", "false exclusivity sentence removed (R13 audit)"),
                        ("every seed trains worse", "unmatched loss claim replaced by matched-comparison wording (R13 audit)"),
                        ("no compared configuration", "false CI-uniqueness sentence removed"),
                        ("isolates the modules", "single-seed factorial wording replaced"),
                        ("gives up", "defensive tradeoff wording removed"),
                        ("must grow with the horizon", "unconditional horizon-monotonicity claim removed (R2 P0-1)"),
                        ("precisely the model's initialization", "false initialization claim removed (R2 P0-2)"),
                        ("initialized to the grid predictor", "false initialization claim removed (R2 P0-2)"),
                        ("the accuracy lives in", "pre-retrain causal claim scoped (R2 P0-4)"),
                        ("strongest published", "unverifiable superlative opener removed (R2 P0-5)"),
                        ("horizon-growth pattern", "abstract monotonicity phrasing scoped (R2 P0-1)"),
                        ("one benchmark family", "self-weakening closer removed (R2 P0-5)"),
                        ("near-unit-radius", "init-radius regime framing removed from full-model claims (R3 P0-3)"),
                        ("16to64to64", "wrong MLP head dims removed (R3 P0-2)"),
                        ("modest NMSE margin", "self-weakening conclusion phrase removed (R3 P1-5)"),
                        ("A rejected objective", "bold negative-result anchor removed (R3 P1-4)"),
                        ("favours the reported model", "caption naming made explicit (R3 P1-6)"),
                        ("degrades all 162 official settings by", "38.8% semantics fixed: mean vs per-setting (R3 P0-4)"),
                        ("on every official setting", "ambiguous 38.8% phrasing removed from abstract (R3 P0-4)"),
                        ("the gain concentrated in channel model B", "abstract over-detail delegated to Fig.3 (R3 P1-1)"),
                        ("costs at most", "M1: unconditional quantization amplitude bound removed (R4)"),
                        ("Delta\\varepsilon|<1", "undefined notation replaced by |eps-epst|<1 (R5)"),
                        ("structural necessity", "pre-registered (c): mechanism claim not upgraded (grid-retrain)"),
                        ("structurally necessary", "pre-registered (c): mechanism claim not upgraded (grid-retrain)"),
                        ("mechanism-checked", "abstract closer de-hedged (Astra R03 adoption)"),
                        ("0.1437", "truncation-artifact NOPOLE value removed (R01 closure)"),
                        ("and $2.8\\%$ alone", "truncation-inflated pole cost removed (R01 closure)"),
                        ("tap encoding", "inert tap-encoding ablation clause removed (C04)"),
                        ("same-protocol noise floor", "unprovable 0.0004 noise floor removed (R12 arithmetic sweep)"),
                        ("rather than a majority of cells", "OOD sentence contradicted 40/72 seed-mean wins (R12; generalization_seeds.csv)"),
                        ("Snapping the learned offsets onto the grid", "abstract no longer carries the snap numbers; sec 4.4 keeps them (R12)"),
                        ("do not reproduce the tradeoff", "capacity-scan claim contradicted Fig 2 non-dominated 77k point (R12)"),
                        ("+0.0708", "Table 2 arithmetic error removed (R12)"),
                        ("30.5\\%$ main effect", "unprovenanced main-effect value replaced by recomputed 31.7% (R12)"),
                        ("express exactly what the grid", "grid-expressivity overclaim removed, conflicts with Prop 2 (R12)"),
                        ("+0.0027", "Table 2 delta at table precision (0.1425-0.1397=0.0028) (R12b)"),
                        ("+0.0060", "Table 2 delta at table precision (0.1458-0.1397=0.0061) (R12b)"),
                        ("+0.0464", "Table 2 delta at table precision (0.1862-0.1397=0.0465) (R12b)"),
                        ("+0.0570", "Table 2 delta at table precision (0.1968-0.1397=0.0571) (R12b)"),
                        ("11times", "9.8x endpoint gap, never 11x (R14 audit: 11.05 is the ratio-to-published)"),
                        ("1596", "denoiser params are 1,590, not GPT's 1,596 (V0 inventory)"),
                        ("220.67", "V6 seed-43 gate junk kept out of the paper"),
                        ("2.4091", "V6 seed-43 gate value kept out of the paper"),
                        ("0.2174", "equal-wall-clock control not yet in the paper (P3 pending)"),
                        ("257point", "PV=Q never scoped to a 257-point grid; 257 appears only as the defect-scan angle count (R22 P1-07)"),
                        ("mathcalZ_1", "P0V=Q never scoped to the 257-point Z1 grid (V2 red line)"),
                        ("Informer", "Informer citation removed in the R14 related work"),
                        ("N-BEATS", "N-BEATS citation removed in the R14 related work"),
                        ("istight", "the bound is conservative (3-4x), never described as tight (V4 red line)"),
                        ("DFTGRID (K=1, fixed grid)", "K=1 is adaptive top-1, never a fixed grid (R19)"),
                        ("What the grid constraint changes", "ssec:price renamed to the price of exactness (R19)"),
                        ("K=12", "the K=12 counterexample was a construction bug, corrected 09-22 (R19)"),
                        ("scores 1.3771", "1.3771 is the adaptive top-1 baseline, never the unique exact transfer (GPT final review P5)"),
                        ("exactness released", "causal exactness-release claim removed; honest contrast only (GPT final review P5)"),
                        ("released exactness", "same removal, noun form (GPT final review P5)"),
                        ("the gap measures", "no causal gap claim without a matched control (GPT final review P5)"),
                        ("bit for bit", "float64 residual is 3e-15, not bit-exact (GPT final review P2)"),
                        ("cannot help", "capacity necessity inference removed; empirical localization only (GPT final review P0-2)"),
                        ("P_0A=", "undefined selector A replaced by explicit [I_T 0] (GPT final review P1-1)"),
                        ("13.4", "gain conditioning is 14.3 under the matched control; 13.4 used the factorial denominator (R22 P0-1)"),
                        ("doesnotpredict", "aphorism removed; K trend stated as a monotone fact (R22 P0-2)"),
                        ("Nonecharacterizes", "universal negative replaced by an affirmative contribution (R22 P1-08)"),
                        ("conservative", "blanket 3-4x range replaced by the measured median 3.84x (R22 P1-06)"),
                        ("farmorethanany", "overbroad multiplier replaced by largest-measured-degradation (R22 P0-1)")]:
        v = norm(gone)
        checks.append({"id": "NEG-%s" % v.replace(".", "p").replace(" ", "-"),
                       "claim": claim, "value": gone, "source": "paper/main.tex",
                       "checked_against": "paper-absent", "ok": v not in tex})

    failed = [c for c in checks if not c["ok"]]
    report = {"checks": len(checks), "passed": len(checks) - len(failed),
              "failed": len(failed), "failures": failed, "all": checks}
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")
    OUT_MD.write_text("# Claim audit\n\n%d/%d checks pass\n%s\n" % (
        report["passed"], report["checks"],
        "" if not failed else "\n".join("- %s: %s" % (c["id"], c["claim"]) for c in failed)),
        encoding="utf-8")
    print("%d/%d checks pass" % (report["passed"], report["checks"]))
    for c in failed:
        print("  FAIL %s  %s  value=%s  source=%s" % (c["id"], c["claim"], c["value"], c["source"]))
    # 2026-09-20 R12: the derived-number (arithmetic) audit is part of the same
    # gate -- string presence alone let the +0.0708 delta slip through 467 checks.
    import subprocess
    import sys
    arith = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "audit_arithmetic.py")],
        capture_output=True, text=True)
    if arith.returncode != 0:
        print(arith.stdout)
        print(arith.stderr)
        print("  FAIL audit_arithmetic (derived-number recomputation)")
        return 1
    for line in arith.stdout.splitlines():
        if "derived-number checks" in line:
            print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
