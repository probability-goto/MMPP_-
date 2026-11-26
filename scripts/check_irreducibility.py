#!/usr/bin/env python
"""論文の数値実験で定常分布を解いた全設定について, 生成行列の既約性を確かめる.

提案モデル (mmpp_predictive.solver) は x0 = (i, s, j, F) = (0, 0, 0, 0) から前向きの幅優先探索で
到達集合 R を求め, 生成行列を R に縮約して GTH で解く. ここでは同じ隣接の定義で向きを逆にした
幅優先探索で, x0 へ到達できる集合 B を求め, R ⊆ B (縮約した行列が既約) かを判定する
(mmpp_experiments.irreducibility). ベースモデルは縮約せずに全状態空間 S で解くので,
R = S かつ S ⊆ B を判定する. あわせて 0 < Q[x, y] < 1e-12 の非対角要素を数える.

設定の一覧 (手で書き写さず, 実験のスクリプトと CSV から作る):
    実験0: scripts/experiment_0_validation.py, scripts/experiment_0P_validation.py --protect の
           既定の引数と build_params (results/experiment_0/*.log の rho と制御パラメータと照合する)
    実験1〜4: results/experiment_{1,2,2_phasewise,3_delta,3_r1,4_short,4_long}.csv の各行
           (mmpp_experiments.design.solve_row と同じ組み立て). 実験4 の内訳 (experiment_4_split.csv)
           は CSV の rho と scripts/experiment_4_K_sensitivity.py の定数 (SPLIT_K, ALWAYS_ON_BETA)
    実験5: results/compare_map/compare_map_rhoB*.csv の各行 (Base, Predictive, Check).
           Check の行で一緒に解いたベースモデルの点と, rho_B=0.8 で実験 8 の CSV から再利用した点の
           元の保護の有無 (scripts/compare_map.py の load_reuse_index) も加える.
           改善の仕組み (scripts/compare_map_mechanism.py) で新しく解くベースモデルの点は,
           その analyze() を実行して design.base_params の呼び出しを記録して得る.
    実験6: results/robustness/robustness_{A,B,BG}.csv の各行

出力:
    results/irreducibility_check.csv      1 行 1 設定
    results/irreducibility_failures.txt   既約でない設定があるときだけ (R \\ B の状態の例)

1 設定ごとに CSV に追記し, 再実行すると確認済みの設定 (source 列) を飛ばして再開する.
最後に設定の一覧の順に並べ直して書き直す.

使用例:
    python scripts/check_irreducibility.py               # 全設定 (2 プロセス)
    python scripts/check_irreducibility.py --jobs 1 --limit 20   # 動作確認
    python scripts/check_irreducibility.py --summary     # CSV から集計だけ表示
    python scripts/check_irreducibility.py --verify-reconstruction 2
        # CSV から組み立て直したパラメータで, CSV ごとに 2 点を解き直して ERP と P_block を照合する
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from mmpp import ModelParameters  # noqa: E402
from mmpp import build_generator as build_base_generator  # noqa: E402
from mmpp.state_space import StateSpace  # noqa: E402
from mmpp_experiments.irreducibility import check_irreducible, offdiag_rates_below  # noqa: E402
from mmpp_experiments.params import SYSTEM, build_mmpp  # noqa: E402
from mmpp_predictive import PredictiveModelParameters  # noqa: E402
from mmpp_predictive import build_generator as build_predictive_generator  # noqa: E402
from mmpp_predictive.state_space import build_is_index, idx_to_state  # noqa: E402

OUT_CSV = os.path.join("results", "irreducibility_check.csv")
FAIL_TXT = os.path.join("results", "irreducibility_failures.txt")
TINY = 1e-12
N_EXAMPLES = 10

PARAM_FIELDS = ["rho_B", "r1", "delta", "alpha", "beta", "n_target", "gamma", "c", "b", "K"]
FIELDS = (["experiment", "model"] + PARAM_FIELDS +
          ["n_states_total", "n_reachable", "n_back_in_R", "irreducible", "n_tiny_rates",
           # 以下は追加の列
           "part", "rho", "sigma", "mu", "protect_presetup", "protect_delayoff",
           "n_backward", "reachable_is_full", "n_scc_reduced", "n_negative_offdiag",
           "min_offdiag_rate", "not_back_examples", "not_reached_examples", "tiny_examples",
           "source"])

# ============================================================
# 設定 (spec): パラメータを組み立てるのに必要な値と, 出所
# ============================================================


def make_spec(experiment, part, model, source, rho, delta, sigma, alpha, beta, c, K, b, mu,
              rho_B=None, r1=None, n_target=None, gamma=None, protect_presetup=None,
              protect_delayoff=None) -> dict:
    rho, delta, sigma = float(rho), float(delta), float(sigma)
    return dict(
        experiment=experiment, part=part, model=model, source=source,
        rho=rho, delta=delta, sigma=sigma, alpha=float(alpha), beta=float(beta),
        c=int(c), K=int(K), b=int(b), mu=float(mu),
        rho_B=rho * (1 + delta) if rho_B in (None, "") else float(rho_B),
        r1="" if r1 in (None, "") else float(r1),
        n_target="" if model == "base" else int(n_target),
        gamma="" if model == "base" else float(gamma),
        protect_presetup="" if model == "base" else bool(protect_presetup),
        protect_delayoff="" if model == "base" else bool(protect_delayoff),
    )


def make_params(spec: dict):
    C0, C1 = build_mmpp(spec["rho"], spec["delta"], spec["sigma"], spec["c"], spec["b"],
                        spec["mu"])
    common = dict(C0=C0, C1=C1, alpha=spec["alpha"], beta=spec["beta"], c=spec["c"],
                  K=spec["K"], b=spec["b"], mu=spec["mu"])
    if spec["model"] == "base":
        return ModelParameters(**common)
    return PredictiveModelParameters(
        **common, n_target=spec["n_target"], gamma=spec["gamma"],
        protect_presetup=spec["protect_presetup"], protect_delayoff=spec["protect_delayoff"])


def _same_params(a, b) -> bool:
    keys = ["c", "K", "b", "mu", "alpha", "beta"]
    if isinstance(a, PredictiveModelParameters):
        keys += ["n_target", "gamma", "protect_presetup", "protect_delayoff",
                 "never_cancel_setup"]
    return (type(a) is type(b) and all(getattr(a, k) == getattr(b, k) for k in keys)
            and np.array_equal(a.C0, b.C0) and np.array_equal(a.C1, b.C1))


def _truthy(v) -> bool:
    return str(v) == "True"


# ============================================================
# 設定の列挙
# ============================================================


def _log_rhos(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return re.findall(r"^=== rho = (\S+?)[ ,]", f.read(), flags=re.M)


def specs_experiment_0() -> List[dict]:
    import experiment_0_validation as e0
    import experiment_0P_validation as e0p

    out = []
    argv = sys.argv
    try:
        sys.argv = ["experiment_0_validation.py"]
        a0 = e0.parse_args()
        sys.argv = ["experiment_0P_validation.py", "--protect"]
        a0p = e0p.parse_args()
    finally:
        sys.argv = argv

    rhos0 = np.linspace(*e0.RHO_RANGE, a0.n_points)
    rhos0p = np.linspace(*e0p.RHO_RANGE, a0p.n_points)
    # 論文の実行記録 (log) の rho と照合する
    for rhos, log in [(rhos0, "base.log"), (rhos0p, "predictive_protect.log")]:
        logged = _log_rhos(os.path.join("results", "experiment_0", log))
        mine = [f"{float(r):.4g}" for r in rhos]
        if logged != mine:
            raise RuntimeError(f"実験0 の rho が {log} と一致しない: {logged} != {mine}")
    with open(os.path.join("results", "experiment_0", "predictive_protect.log"),
              encoding="utf-8") as f:
        heads = set(re.findall(r"n_target=(\d+), gamma=(\S+), protect=(\w+) ===", f.read()))
    if heads != {(str(a0p.n_target), str(a0p.gamma), "True")}:
        raise RuntimeError(f"実験0-P の制御パラメータが log と一致しない: {heads}")

    for rho in rhos0:
        p = e0.build_params(float(rho))
        s = make_spec("0", "base", "base", f"scripts/experiment_0_validation.py:rho={float(rho)!r}",
                      rho, e0.DELTA, e0.SIGMA, p.alpha, p.beta, p.c, p.K, p.b, p.mu,
                      r1=p.alpha / e0.SIGMA)
        assert _same_params(make_params(s), p)
        out.append(s)
    for rho in rhos0p:
        p = e0p.build_params(float(rho), a0p.n_target, a0p.gamma, a0p.protect)
        s = make_spec("0", "predictive", "predictive",
                      f"scripts/experiment_0P_validation.py --protect:rho={float(rho)!r}",
                      rho, e0p.DELTA, e0p.SIGMA, p.alpha, p.beta, p.c, p.K, p.b, p.mu,
                      r1=p.alpha / e0p.SIGMA, n_target=p.n_target, gamma=p.gamma, protect_presetup=p.protect_presetup,
                      protect_delayoff=p.protect_delayoff)
        assert _same_params(make_params(s), p)
        out.append(s)
    return out


def _read(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def specs_design_csv(experiment: str, part: str, path: str) -> List[dict]:
    """mmpp_experiments.design.solve_row で作った CSV (ベースモデル) の各行."""
    out = []
    for n, r in enumerate(_read(path), start=2):
        out.append(make_spec(experiment, part, "base", f"{path}:line={n}", r["rho"], r["delta"],
                             r["sigma"], r["alpha"], r["beta"], SYSTEM["c"], r["K"], SYSTEM["b"],
                             SYSTEM["mu"], rho_B=r["rho_B"], r1=r["r1"]))
    return out


def specs_experiment_4_split() -> List[dict]:
    import experiment_4_K_sensitivity as e4
    from mmpp_experiments import design as d

    path = os.path.join("results", "experiment_4_split.csv")
    out = []
    for n, r in enumerate(_read(path), start=2):
        out.append(make_spec("4", "split", "base", f"{path}:line={n} (always_on)", r["rho"],
                             d.DELTA, d.sigma_of(d.R1_LONG), d.ALPHA, e4.ALWAYS_ON_BETA,
                             SYSTEM["c"], e4.SPLIT_K, SYSTEM["b"], SYSTEM["mu"],
                             r1=d.R1_LONG))
    return out


def _spec_from_map_row(experiment, part, source, r, model=None, protect=None) -> dict:
    model = r["model"] if model is None else model
    pp = _truthy(r["protect_presetup"]) if protect is None else protect[0]
    pd = _truthy(r["protect_delayoff"]) if protect is None else protect[1]
    return make_spec(experiment, part, model, source, r["rho"], r["delta"], r["sigma"],
                     r["alpha"], r["beta"], r["c"], r["K"], r["b"], r["mu"], rho_B=r["rho_B"],
                     r1=r["r1"], n_target=r["n_target"], gamma=r["gamma"],
                     protect_presetup=pp, protect_delayoff=pd)


def specs_compare_map() -> List[dict]:
    import compare_map as cm

    out = []
    for rho_B in cm.RHO_B_LEVELS:
        path = cm.csv_path(rho_B)
        reuse = cm.load_reuse_index(rho_B)
        for n, r in enumerate(_read(path), start=2):
            src = f"{path}:line={n}"
            out.append(_spec_from_map_row("5", r["series"], src, r))
            if r["series"] == "Check":
                # 整合性の検査では同じ beta のベースモデルも解いている (CSV には Predictive の行だけ)
                out.append(_spec_from_map_row("5", "Check(base)", src + " (check: base)", r,
                                              model="base"))
            if r["source"].startswith("reused:"):
                orig = reuse[cm.rkey(r)]
                flags = (_truthy(orig["protect_presetup"]), _truthy(orig["protect_delayoff"]))
                if r["model"] == "predictive" and flags != (_truthy(r["protect_presetup"]),
                                                            _truthy(r["protect_delayoff"])):
                    # 再利用の元 (実験 8) で実際に解いた保護の有無
                    out.append(_spec_from_map_row(
                        "5", "Predictive(reused, original flags)",
                        src + f" (original flags in {r['source'][7:]})", r, protect=flags))
    return out


def specs_compare_map_mechanism() -> List[dict]:
    """compare_map_mechanism.analyze() を実行し, ベースモデルを組み立てた呼び出しを記録する."""
    import compare_map_mechanism as mech
    from mmpp_experiments import design as d

    calls = []
    original = d.base_params

    def recording(rho, delta, sigma, alpha=d.ALPHA, beta=None, K=SYSTEM["K"]):
        p = original(rho, delta, sigma, alpha, beta, K)
        calls.append((dict(rho=rho, delta=delta, sigma=sigma, alpha=alpha, beta=p.beta, K=K), p))
        return p

    d.base_params = recording
    try:
        mech.analyze()
    finally:
        d.base_params = original

    out, seen = [], {}
    for k, (a, p) in enumerate(calls):
        key = tuple(sorted(a.items()))
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        s = make_spec("5", "mechanism", "base", f"scripts/compare_map_mechanism.py:call={k}",
                      a["rho"], a["delta"], a["sigma"], a["alpha"], a["beta"], SYSTEM["c"],
                      a["K"], SYSTEM["b"], SYSTEM["mu"],
                      r1=a["alpha"] / a["sigma"] if a["delta"] > 0 else "")
        assert _same_params(make_params(s), p)
        out.append(s)
    print(f"実験5 (改善の仕組み): base_params の呼び出し {len(calls)} 回, 異なる設定 {len(out)}")
    return out


def specs_robustness() -> List[dict]:
    out = []
    for part in ["A", "B", "BG"]:
        path = os.path.join("results", "robustness", f"robustness_{part}.csv")
        for n, r in enumerate(_read(path), start=2):
            out.append(_spec_from_map_row("6", part, f"{path}:line={n}", r))
    return out


def all_specs() -> List[dict]:
    specs = specs_experiment_0()
    specs += specs_design_csv("1", "", os.path.join("results", "experiment_1.csv"))
    specs += specs_design_csv("2", "", os.path.join("results", "experiment_2.csv"))
    specs += specs_design_csv("2", "phasewise", os.path.join("results", "experiment_2_phasewise.csv"))
    specs += specs_design_csv("3", "delta", os.path.join("results", "experiment_3_delta.csv"))
    specs += specs_design_csv("3", "r1", os.path.join("results", "experiment_3_r1.csv"))
    specs += specs_design_csv("4", "short", os.path.join("results", "experiment_4_short.csv"))
    specs += specs_design_csv("4", "long", os.path.join("results", "experiment_4_long.csv"))
    specs += specs_experiment_4_split()
    specs += specs_compare_map()
    specs += specs_compare_map_mechanism()
    specs += specs_robustness()
    sources = [s["source"] for s in specs]
    if len(set(sources)) != len(sources):
        raise RuntimeError("source が重複している")
    return specs


# ============================================================
# 1 設定の確認
# ============================================================


def _decoder(spec: dict, params):
    if spec["model"] == "base":
        ss = StateSpace(params)
        return lambda k: ss.decode(int(k))
    _, pairs = build_is_index(params.c)
    return lambda k: idx_to_state(int(k), params.c, params.K, params.D_M, pairs)


def check_one(spec: dict) -> dict:
    params = make_params(spec)
    Q = (build_base_generator(params) if spec["model"] == "base"
         else build_predictive_generator(params))
    res = check_irreducible(Q, 0)
    rows, cols, vals = offdiag_rates_below(Q, TINY)
    decode = _decoder(spec, params)
    off = Q.tocoo()
    off_vals = off.data[(off.row != off.col) & (off.data > 0)]

    irreducible = res.irreducible and (res.full if spec["model"] == "base" else True)
    not_reached = (np.setdiff1d(np.arange(res.n_states), res.reachable)
                   if spec["model"] == "base" else np.array([], dtype=int))
    out = {k: spec[k] for k in ["experiment", "model", "part", "rho", "sigma", "mu",
                                 "protect_presetup", "protect_delayoff", "source"] + PARAM_FIELDS}
    out.update(
        n_states_total=res.n_states, n_reachable=res.n_reachable, n_back_in_R=res.n_back_in_R,
        irreducible=irreducible, n_tiny_rates=len(vals),
        n_backward=len(res.backward), reachable_is_full=res.full,
        n_scc_reduced=res.n_scc_reduced, n_negative_offdiag=res.n_negative_offdiag,
        min_offdiag_rate=float(off_vals.min()),
        not_back_examples=" ".join(str(decode(k)).replace(" ", "")
                                   for k in res.not_back[:N_EXAMPLES]),
        not_reached_examples=" ".join(str(decode(k)).replace(" ", "")
                                      for k in not_reached[:N_EXAMPLES]),
        tiny_examples=" ".join(f"{decode(r)}->{decode(c)}={v:.3g}".replace(" ", "")
                               for r, c, v in list(zip(rows, cols, vals))[:N_EXAMPLES]),
    )
    return out


# ============================================================
# 実行と集計
# ============================================================


def _done_sources(path: str) -> set:
    if not os.path.exists(path):
        return set()
    return {r["source"] for r in _read(path)}


def run(specs: List[dict], jobs: int, out_csv: str) -> None:
    done = _done_sources(out_csv)
    todo = [s for s in specs if s["source"] not in done]
    print(f"設定 {len(specs)} (確認済み {len(specs) - len(todo)}, 残り {len(todo)})", flush=True)
    new_file = not os.path.exists(out_csv)
    t0 = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        it = (map(check_one, todo) if jobs == 1
              else Pool(jobs).imap_unordered(check_one, todo, chunksize=4))
        for k, row in enumerate(it, start=1):
            w.writerow(row)
            f.flush()
            if not row["irreducible"] or row["n_tiny_rates"]:
                print(f"  [要確認] {row['source']}: irreducible={row['irreducible']} "
                      f"n_tiny_rates={row['n_tiny_rates']}", flush=True)
            if k % 200 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)} ({el:.0f} s, 残り約 {el / k * (len(todo) - k):.0f} s)",
                      flush=True)


def finalize(specs: List[dict], out_csv: str) -> List[dict]:
    """設定の一覧の順に並べ直して書き直す. 一覧にない行 (古い行) は捨てる."""
    by_src = {r["source"]: r for r in _read(out_csv)}
    missing = [s["source"] for s in specs if s["source"] not in by_src]
    rows = [by_src[s["source"]] for s in specs if s["source"] in by_src]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if missing:
        print(f"確認できなかった設定: {len(missing)}")
        for m in missing:
            print(f"  {m}")
    return rows


def summarize(rows: List[dict], fail_txt: str, n_specs: Optional[int] = None) -> None:
    bad = [r for r in rows if r["irreducible"] != "True"]
    tiny = [r for r in rows if int(r["n_tiny_rates"]) > 0]
    neg = [r for r in rows if int(r["n_negative_offdiag"]) > 0]
    scc = [r for r in rows if r["model"] == "predictive" and r["n_scc_reduced"] != "1"]
    print("=" * 60)
    if n_specs is not None:
        print(f"列挙した設定数: {n_specs}")
    print(f"全設定数: {len(rows)}")
    by: Dict[tuple, List[int]] = {}
    for r in rows:
        v = by.setdefault((r["experiment"], r["model"]), [0, 0, 0])
        v[0] += 1
        v[1] += r["irreducible"] != "True"
        v[2] += int(r["n_tiny_rates"]) > 0
    for (e, m), (n, nb, nt) in sorted(by.items()):
        print(f"  実験{e} {m:<10}: {n:5d} 設定, 既約でない {nb}, 微小な遷移率あり {nt}")
    print(f"既約でない設定数: {len(bad)}")
    print(f"微小な遷移率 (0 < q < {TINY:g}) を含む設定数: {len(tiny)}")
    print(f"負の非対角要素を含む設定数: {len(neg)}")
    print(f"縮約した行列の強連結成分が 1 でない提案モデルの設定数: {len(scc)}")
    mins = [float(r["min_offdiag_rate"]) for r in rows]
    if mins:
        k = int(np.argmin(mins))
        print(f"非対角の正の遷移率の最小値: {mins[k]:.3g} ({rows[k]['source']})")
    for r in tiny[:N_EXAMPLES]:
        print(f"  微小な遷移率: {r['source']}: {r['n_tiny_rates']} 件, 例 {r['tiny_examples']}")
    if bad:
        with open(fail_txt, "w", encoding="utf-8") as f:
            for r in bad:
                f.write(f"{r['source']}\n  experiment={r['experiment']} model={r['model']} "
                        + " ".join(f"{k}={r[k]}" for k in PARAM_FIELDS)
                        + f"\n  n_states_total={r['n_states_total']} n_reachable={r['n_reachable']}"
                        f" n_back_in_R={r['n_back_in_R']}"
                        f" |R\\B|={int(r['n_reachable']) - int(r['n_back_in_R'])}\n"
                        f"  R\\B の例: {r['not_back_examples']}\n"
                        f"  S\\R の例 (ベースモデル): {r['not_reached_examples']}\n")
        print(f"既約でない設定を {fail_txt} に書いた")
    elif os.path.exists(fail_txt):
        os.remove(fail_txt)


def verify_reconstruction(specs: List[dict], per_file: int) -> None:
    """CSV ごとに per_file 点を, 組み立て直したパラメータで解き直し, CSV の ERP, P_block と照合する."""
    from mmpp_experiments.evaluate import (base_metrics, predictive_metrics, solve_base,
                                           solve_predictive)

    by_file: Dict[str, List[dict]] = {}
    for s in specs:
        m = re.match(r"(results/\S+\.csv):line=(\d+)$", s["source"])
        if m and "split" not in s["source"]:
            by_file.setdefault(m.group(1), []).append(s)
    worst = 0.0
    for path, ss in by_file.items():
        rows = _read(path)
        picks = [ss[int(round(x))] for x in np.linspace(0, len(ss) - 1, per_file)]
        for s in picks:
            r = rows[int(s["source"].rsplit("=", 1)[1]) - 2]
            p = make_params(s)
            got = (base_metrics(solve_base(p)) if s["model"] == "base"
                   else predictive_metrics(solve_predictive(p)))
            rel = max(abs(got[k] / float(r[k]) - 1) for k in ["ERP", "P_block_arrival_stable"])
            worst = max(worst, rel)
            print(f"  {s['source']}: ERP, P_block の最大相対差 {rel:.2e}", flush=True)
    print(f"組み立て直しの照合: 最大相対差 {worst:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="先頭のこの数の設定だけ (動作確認用)")
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--fail-out", default=FAIL_TXT)
    parser.add_argument("--summary", action="store_true", help="CSV から集計だけ表示する")
    parser.add_argument("--verify-reconstruction", type=int, default=0, metavar="N")
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.summary:
        summarize(_read(args.out), args.fail_out)
        return
    specs = all_specs()
    if args.limit is not None:
        specs = specs[:args.limit]
    if args.verify_reconstruction:
        verify_reconstruction(specs, args.verify_reconstruction)
        return
    run(specs, args.jobs, args.out)
    summarize(finalize(specs, args.out), args.fail_out, len(specs))


if __name__ == "__main__":
    main()
