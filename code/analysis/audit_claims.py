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
    """Read a derived table, falling back to the working repository.

    The anonymous package carries its own copy of results/analysis/, but the
    tables added on 2026-09-17 (three-seed shift, loss-off ablation, A/C/D
    seeds, pole intervention, mobility bands, classical ladder) only exist in
    the working repository until the writer syncs them. Falling back keeps this
    audit runnable in both places; a genuinely missing file yields no checks
    instead of a crash, and the summary line makes that visible.
    """
    path = ANA / name
    if not path.exists():
        fallback = ROOT.parent / "results" / "analysis" / name
        if fallback.exists():
            path = fallback
        else:
            print("  note: %s not found in this package (skipped)" % name)
            return []
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
                fallback = ROOT.parent / "results" / "analysis" / source
                path = fallback if fallback.exists() else None
            ok = v in norm(path.read_text(encoding="utf-8") if path else source)
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
                          "timemixer-L-iclr24", "dlinear-baseline", "tcn-baseline"}
        # 2026-09-19 round 2: the rejected-objective paragraph no longer quotes
        # the with-loss NMSEs; the rows stay regenerable evidence.
        withloss = label.startswith("ours-with-loss")
        w = "evidence" if (cap or prose or withloss) else "paper"
        wn = "evidence" if (cap or withloss) else "paper"
        check("T1-%d-nmse" % i, "Table 1 %s NMSE" % label, r["nmse"], "main_table.csv", where=wn)
        if r.get("params") and r["params"] not in ("", "None"):
            check("T1-%d-params" % i, "Table 1 %s params" % label, r["params"], "main_table.csv", where=w)
        if (r.get("macs_per_sample") and r["macs_per_sample"] not in ("", "None")
                and w == "paper"):  # fmt_macs string only exists in the paper
            check("T1-%d-macs" % i, "Table 1 %s MACs" % label,
                  fmt_macs(r["macs_per_sample"]), "main_table.csv", where=w)

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
        check("EFF-%s-params" % label, "profile %s params" % label, params, prof, where="evidence")
        check("EFF-%s-macs" % label, "profile %s MACs" % label, macs, prof, where="evidence")
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
                     "0.2300"}  # 1.8081 (no prediction) retired from Table 1
    for i, r in enumerate(read_csv("classical_ladder.csv"), 1):
        where = "paper" if r["cell_mean_nmse"] in QUOTED_LADDER else "evidence"
        check("LADDER-%d" % i, "classical ladder %s/%s" % (r["split"], r["model"]),
              r["cell_mean_nmse"], "classical_ladder.csv", where=where)

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
        where = "paper" if r.get("pct_vs_stock", r.get("pct")) in ("37.0", "38.8") else "evidence"
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
            check("TH-h%s" % r["value"], "grid-snap penalty at horizon %s (Theorem 1 closure)" % r["value"],
                  r["pct"], "pole_snap_horizon.csv")
    # Offset quantisation curve: 1/8 / 1/4 / 1/2-bin penalties.
    for r in read_csv("pole_intervention_curve.csv"):
        if r.get("family") == "quant":
            check("TH-quant-%s" % r["tag"], "offset quantisation penalty (%s bin)" % r["tag"],
                  r["pct_vs_stock"], "pole_intervention_curve.csv")
    # Concept words the reframing depends on.
    for word, claim in [("semigroup", "local semigroup statement present"),
                        ("canonical", "canonical readout statement present"),
                        ("Theorem~1", "Theorem 1 referenced"),
                        ("varepsilon", "total offset notation present")]:
        check("TH-word-%s" % word, claim, word, "paper/main.tex")
    flat_tex = " ".join(raw_tex.split())  # line-wrapping-insensitive phrase matching
    # 2026-09-19 round 2: scoped-theory and warm-start provenance language.
    invariants_r2 = [("R2-phase", "h-dependent phase language present (scoped theorem claim)",
                      "dependent phase" in raw_tex),
                     ("R2-ridge", "ridge least-squares spectral warm start stated (R4 wording)",
                      "ridge least-squares spectral warm start" in flat_tex),
                     ("R2-frozen", "frozen-predictor causal scoping present",
                      "relies on its learned off-grid placement" in raw_tex),
                     ("R2-limit", "grid predictor framed as limit, not initialization",
                      "unit-radius, zero-offset limit" in raw_tex)]
    for cid, claim, ok in invariants_r2:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-19 round 3: implementation-spec accuracy and artifact compliance.
    invariants_r3 = [("R3-mlp32", "pole head dims 16->64->32 stated (R3 P0-2)",
                      "16{\\to}64{\\to}32" in flat_tex),
                     ("R3-mlp128", "gain head dims 16->64->128 stated (R3 P0-2)",
                      "16{\\to}64{\\to}128" in flat_tex),
                     ("R3-radius", "snap narrative acknowledges learned radius attenuation (R3 P0-3)",
                      "despite radius attenuation" in flat_tex),
                     ("R3-phase-mono", "phase-factor monotonicity scoped to |deps|<1, h<=4 (R3 P0-3)",
                      "phase factor of Corollary~1 is monotone in $h$" in flat_tex and "near-unit-radius" not in flat_tex),
                     ("R3-abstract-split", "38.8% per-setting scope carried by sec 4.4 (R3 P0-4; R12 abstract slim-down)",
                      "with all 162 settings worse" in flat_tex),
                     ("R3-den", "denoiser included in parameter attribution (R3 P1-3)",
                      "the operator, the residual denoiser" in flat_tex),
                     ("R3-kc", "K_c framed as poles-only core, gains in readout (R3 P1-2)",
                      "whose poles and per-mode readout gains are generated from the CSI history" in flat_tex),
                     ("R3-caption", "pareto caption states the three-seed-validated operating point (R3 P1-6; R12 scan-claim fix)",
                      "the reported model is the three-seed-validated operating" in flat_tex),
                     ("R3-artifact", "figure shipped as Inkscape vector PDF, no raster (Type 3 compliance, R3 P0-1 + R5 vector swap)",
                      (ROOT / "paper" / "fig1_final.pdf").exists()
                      and not (ROOT / "paper" / "fig1_final.png").exists())]
    for cid, claim, ok in invariants_r3:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-19 round 4: M1 bound removal + measured/consistent abstract + warm-start precision.
    invariants_r4 = [("R4-abstract-measured", "horizon growth of the snap penalty stated as a measured series in sec 4.4 (R4; R12 moved out of abstract)",
                      "the aggregate penalty rises monotonically despite radius attenuation" in flat_tex),
                     ("R4-warmstart-precise", "warm start described as ridge spectral fit on the grid (R4 P1-2)",
                      "ridge least-squares spectral warm start on the grid" in flat_tex
                      and "ridge-warm-started from a spectral fit on the grid" in flat_tex)]
    for cid, claim, ok in invariants_r4:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-20 grid-retrain consumption: pre-registered outcome (c), GRID_RETRAIN_BRIEF_2026-09-18 sec.3.
    # Provenance: results/analysis/grid_retrain_summary.csv, results/analysis/abloff_shift_attribution.csv,
    # results/run_csv/GRIDFIX_S{42,43,44}_{full162,gen_gen}.csv, results/run_csv/ABLOFF_*_gen_gen.csv.
    invariants_r6 = [("R6-gridfix-numbers", "grid-pinned retrain control: both slices + paired CI + three seeds",
                      "0.1401 against 0.1402" in flat_tex
                      and "0.2688 against 0.2716 unseen, paired CI $[-0.0037,-0.0020]$, three seeds)" in flat_tex),
                     ("R6-punchline", "solution-level reliance distinguished from architectural need (Astra sec.9 adoption)",
                      "the calibrated transfer carries the performance; frozen snapping measures how strongly the fitted parameterization leans on the off-grid freedom" in flat_tex),
                     ("R6-conclusion-scoped", "reliance statement stays in sec 4.4; conclusion ends on the positive frame (Astra sec.7.6)",
                      "The frozen predictor's shift advantage therefore relies on its learned off-grid placement" in flat_tex),
                     ("R6-attrib-denoiser", "unseen-slice shift advantage attributed to denoiser with CI",
                      "$+20.9\\%$ without it, 95\\% CI $[+0.0431,+0.0713]$" in flat_tex),
                     ("R6-attrib-decomp-pole", "decomposition secondary; pole removal slightly helps on unseen slice",
                      "to the delay decomposition ($+4.0\\%$); removing the pole conditioning slightly helps ($-0.9\\%$)" in flat_tex)]
    for cid, claim, ok in invariants_r6:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-20 Astra round consumption: retitle + fused abstract + error-chain theory +
    # Table-1 regroup + fixed-grid row. Provenance: results/analysis/ablation_table_off.csv
    # (R01 closure: full 162-row ABLOFF_NOPOLE), results/analysis/abloff_shift_attribution.csv,
    # DERIVATION_PACKAGE.md (A2/A5, hand-re-derived), make_fig_pareto.py frontier().
    invariants_r7 = [("R7-title", "title reframed to compact spectral transfer (Astra R03/R04)",
                      "DD-Koopman: Compact Spectral Transfer for Multi-Step CSI Prediction" in flat_tex),
                     ("R7-abstract-opener", "abstract opens on the accuracy-cost tradeoff, not a generic-field claim",
                      "must balance forecasting accuracy against inference cost" in flat_tex),
                     ("R7-abstract-theory", "abstract carries the error-chain sentence",
                      "separates input cleaning, spectral propagation and readout calibration" in flat_tex),
                     ("R7-abstract-closer", "abstract closer: structured spectral parameterization (R7; R12 reviewer adoption)",
                      "with a structured spectral parameterization" in flat_tex),
                     ("R7-prop2", "fixed-grid expressivity proposition grounds the retrain control",
                      "spans every shared linear multi-step predictor" in flat_tex
                      and "Proposition 2 (fixed-grid expressivity)" in flat_tex),
                     ("R7-errchain", "error-chain theorem names the three design roles",
                      "input cleaning controls $e_Z$, pole placement controls $e_K$, and readout calibration controls $e_G$" in flat_tex),
                     ("R7-corollary", "sensitivity identity demoted to Corollary 1",
                      "Corollary 1 (single-mode phase law)" in flat_tex
                      and "phase factor of Corollary~1" in flat_tex),
                     ("R7-factorial-pole", "factorial pole cell at corrected full-grid value (R01 closure; R10 matrix form)",
                      "0.1408 & $+0.0011$" in flat_tex),
                     ("R7-notap", "inert tap-encoding clause removed per C04",
                      "One implementation choice is not load-bearing" in flat_tex),
                     ("R7-table-retrained", "fixed-grid retrained row visible in Table 1 (Astra R04)",
                      "ours, fixed-grid retrained (3 seeds)" in flat_tex),
                     ("R7-caption-macs", "MACs per row moved to caption footnote",
                      "804.9M (reference), 6.51M" in flat_tex),
                     ("R7-pareto-caption", "pareto caption documents non-dominated marking",
                      "The step line connects the non-dominated points" in flat_tex),
                     ("R7-fig1-caption", "fig 1 caption names the specific reference, not the field (Astra sec.8)",
                      "The 21.9M-parameter CSI-4CAST reference attaches the delay domain" in flat_tex),
                     ("R7-conclusion-chain", "conclusion: analysis/ablation roles separated, no claim that ablations validate the bound (R7; R12 reviewer adoption)",
                      "The error analysis separates input, propagation and readout contributions" in flat_tex),
                     ("R7-intro-cleans-first", "intro states the denoiser-first design",
                      "DD-Koopman cleans the observation first and learns all three from it" in flat_tex)]
    for cid, claim, ok in invariants_r7:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-20 R9: Table 2 (tab:factorial) restores the ablation table as a
    # complete 2^3 under one protocol (240 ep, seed 42); numbers move from
    # prose to cells, interpretation sentences stay in the text.
    invariants_r9 = [("R9-tab2-complete", "Table 2 shows the full 2^3 incl. the decomp+pole pair row",
                      "0.1458 & $+0.0061$" in flat_tex),
                     ("R9-tab2-protocol", "Table 2 caption discloses the factorial protocol",
                      "spectral loss off, 240 epochs, seed 42" in flat_tex),
                     ("R9-tab2-pointer", "prose references Table 2 for the factorial",
                      "Table~\\ref{tab:factorial} runs the full $2^3$" in flat_tex),
                     ("R9-tab2-interpretation", "superadditivity interpretation survives the tabulation",
                      "costs more than the sum of the parts" in flat_tex),
                     ("R9-tab2-warmstart-prose", "warm start stays an implementation choice in prose",
                      "removing the least-squares warm start costs $0.9\\%$" in flat_tex)]
    for cid, claim, ok in invariants_r9:
        checks.append({"id": cid, "claim": claim, "value": "-", "source": "paper/main.tex",
                       "checked_against": "paper", "ok": bool(ok)})
    # 2026-09-20 R12: external-review adoption + zero-tolerance arithmetic sweep
    # (audit_arithmetic.py). Table 2 delta +0.0708 -> +0.0464 (ablation_table_off.csv
    # delta_vs_reference), Table 1 gridfix x 1.13 -> 1.12 (0.1401/0.1246, both
    # rounding conventions), denoiser main effect +30.5% -> +31.7% (mean of the
    # four flip-pair deltas 0.0382/0.0543/0.0445/0.0404 over ref 0.1397; no CSV
    # carried 30.5), unprovable noise floor 0.0004 removed, capacity-scan claim
    # corrected (77k scan point 0.1367 < 0.1402; variant_family_comparison.csv
    # 77k-vs-173k CI spans zero), off-grid narrative rebalanced out of the
    # abstract/conclusion (fixed-grid retrain 0.1401 vs 0.1402 is in Table 1),
    # OOD "majority of cells" sentence removed (all-seedmean favour_cells 40/72,
    # generalization_seeds.csv), Corollary 1 common-radius hypothesis added,
    # Proposition 1 exact-recovery/stability split.
    invariants_r12 = [("R12-tab2-delta", "Table 2 all-three-off delta equals table-precision arithmetic (0.1862-0.1397)",
                       "0.1862 & $+0.0465$" in flat_tex),
                      ("R12-tab1-gridfix-x", "Table 1 gridfix ratio recomputed from displayed NMSEs",
                       "0.1401 & 1.12" in flat_tex),
                      ("R12-main-effect", "denoiser main effect recomputed from the factorial",
                       "$+31.7\\%$ average main effect across the four flip pairs" in flat_tex),
                      ("R12-capacity-claim", "capacity scan stated as competitive, not non-reproducing (reviewer R02)",
                       "the width scan holds competitive accuracy from 58k to 327k parameters" in flat_tex),
                      ("R12-alt-arch-range", "alternative architectures scoped to the accuracy range",
                       "do not reach this range" in flat_tex
                       and "do not reach its accuracy" in flat_tex),
                      ("R12-corollary-radius", "Corollary 1 states the common-radius hypothesis (reviewer R04a)",
                       "share the radius $r$" in flat_tex),
                      ("R12-prop1-stability", "Prop 1 separates stability from exact recovery (reviewer R04b)",
                       "stability, not the exact recovery" in flat_tex),
                      ("R12-structured-param", "off-grid sentence states parameterization + active use (reviewer R04c)",
                       "structured parameterization of\nthe finite-window spectral transfer" in tex
                       or "structured parameterization of the finite-window spectral transfer" in flat_tex),
                      ("R12-conclusion-sensitivity", "conclusion carries the sensitivity statement",
                       "reveal the strong sensitivity of the fitted off-grid parameterization" in flat_tex)]
    for cid, claim, ok in invariants_r12:
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
                        ("+0.0570", "Table 2 delta at table precision (0.1968-0.1397=0.0571) (R12b)")]:
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
