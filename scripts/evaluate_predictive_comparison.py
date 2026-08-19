#!/usr/bin/env python
"""実験 1-P〜4-P vs 実験 1〜4 の体系的な比較評価.

各実験の CSV (ベースと Predictive) を統合的に読み込み, 以下を Markdown
形式で標準出力に出す:

1. 全体サマリ (総データ点数, 勝率, 平均改善率)
2. 実験別サマリ (実験 1-P, 2-P, 3-P, 4-P それぞれで勝敗と改善幅)
3. 条件別詳細 (バースト水準 x alpha x 主要制御変数の各条件で 4 指標を評価)
4. パターン抽出 (Predictive が強い条件・弱い条件の特定)

注意: ベース実験 (1〜4) は少数点の粗いグリッド (rho/beta/delta/sigma
それぞれ 3〜5 点), Predictive 実験 (1-P〜4-P) は密なグリッド (15〜40点)
で実行されている. そのため制御変数の完全一致ではなく, 連続変数については
最近傍点マッチング (beta, sigma は対数距離, rho, delta は線形距離) で
対応付けている. alpha, K, burst_name のような離散変数は厳密一致を要求する.

使用例:
    python scripts/evaluate_predictive_comparison.py > evaluation_report.md
    python scripts/evaluate_predictive_comparison.py --n-target 10 --gamma 5.0
"""
import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

METRICS = ["P_block_arrival", "E_W", "Cost", "ERP"]


@dataclass
class ComparisonPoint:
    """1 データ点でのベース vs Predictive 比較."""

    experiment: str  # "1P", "2P-medium", "3P-delta", "4P-weak" など
    burst_name: str
    control_vars: Dict[str, float]
    metric: str
    base_value: float
    pred_value: float
    match_error: float = 0.0  # 最近傍マッチングの誤差 (相対 or 絶対)

    @property
    def improvement_pct(self) -> float:
        """改善率 (%). 正: Predictive が良い, 負: Predictive が悪い."""
        if abs(self.base_value) < 1e-12:
            return 0.0
        return (self.base_value - self.pred_value) / self.base_value * 100

    @property
    def is_predictive_better(self) -> bool:
        return self.improvement_pct > 0

    @property
    def is_predictive_much_better(self) -> bool:
        return self.improvement_pct > 20

    @property
    def is_predictive_much_worse(self) -> bool:
        return self.improvement_pct < -20


def load_csv(filepath: str) -> List[Dict]:
    """CSV を辞書リストとして読み込む (数値列は float に変換)."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for k in list(row.keys()):
                if k in ("burst_name", "sweep_var"):
                    continue
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def _field_eq(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if a is None or b is None:
        return False
    return abs(a - b) < tol


def match_rows(
    base_rows: List[Dict],
    pred_rows: List[Dict],
    exact_fields: List[str],
    nearest_field: Optional[str] = None,
    log_scale: bool = False,
) -> List[Tuple[Dict, Dict, float]]:
    """ベースと Predictive の CSV 行を対応付ける.

    exact_fields は厳密一致 (離散変数: alpha, K, burst_name 等).
    nearest_field は最近傍マッチング (連続変数: rho, beta, delta, sigma).
    ベースは粗いグリッド, Predictive は密なグリッドのため, 連続変数は
    完全一致しないことが多い.
    """
    pairs = []
    for b in base_rows:
        candidates = [
            p for p in pred_rows
            if all(_field_eq(b.get(f), p.get(f)) for f in exact_fields)
        ]
        if not candidates:
            continue
        if nearest_field is None:
            best = candidates[0]
            err = 0.0
        else:
            bv = b.get(nearest_field)
            if bv is None:
                continue

            def dist(p: Dict) -> float:
                pv = p.get(nearest_field)
                if pv is None:
                    return float("inf")
                if log_scale and bv > 0 and pv > 0:
                    return abs(math.log(bv) - math.log(pv))
                return abs(bv - pv)

            best = min(candidates, key=dist)
            best_pv = best.get(nearest_field)
            if log_scale and bv > 0 and best_pv and best_pv > 0:
                err = abs(best_pv - bv) / bv
            else:
                err = abs(best_pv - bv) if best_pv is not None else float("inf")
        pairs.append((b, best, err))
    return pairs


def build_comparison_points(
    experiment: str,
    base_rows: List[Dict],
    pred_rows: List[Dict],
    exact_fields: List[str],
    control_var_fields: List[str],
    nearest_field: Optional[str] = None,
    log_scale: bool = False,
    metrics: Optional[List[str]] = None,
) -> List[ComparisonPoint]:
    """マッチしたペアから ComparisonPoint のリストを生成."""
    if metrics is None:
        metrics = METRICS
    points = []
    triples = match_rows(base_rows, pred_rows, exact_fields, nearest_field, log_scale)
    for base_row, pred_row, err in triples:
        burst_name = base_row.get("burst_name", "")
        if not isinstance(burst_name, str):
            burst_name = str(burst_name)
        control_vars = {f: base_row[f] for f in control_var_fields if f in base_row}
        for metric in metrics:
            if metric not in base_row or metric not in pred_row:
                continue
            points.append(ComparisonPoint(
                experiment=experiment,
                burst_name=burst_name,
                control_vars=control_vars,
                metric=metric,
                base_value=base_row[metric],
                pred_value=pred_row[metric],
                match_error=err,
            ))
    return points


def load_experiment_1p(results_dir: str, n_target: int, gamma: float) -> List[ComparisonPoint]:
    """実験 1 vs 1-P の比較点を生成 (burst 別 rho スイープ)."""
    base_rows = load_csv(os.path.join(results_dir, "experiment_1.csv"))
    pred_rows = load_csv(os.path.join(results_dir, f"experiment_1P_nt{n_target}_g{gamma}.csv"))
    return build_comparison_points(
        "1P", base_rows, pred_rows,
        exact_fields=["burst_name"],
        control_var_fields=["burst_name", "rho"],
        nearest_field="rho", log_scale=False,
    )


def load_experiment_2p(
    results_dir: str, n_target: int, gamma: float, burst_name: str,
) -> List[ComparisonPoint]:
    """実験 2 vs 2-P の比較点を生成 (バースト別, alpha 厳密一致 x beta 対数最近傍)."""
    base_rows = load_csv(os.path.join(results_dir, f"experiment_2_{burst_name}.csv"))
    pred_rows = load_csv(os.path.join(
        results_dir, f"experiment_2P_{burst_name}_nt{n_target}_g{gamma}.csv"
    ))
    if not base_rows:
        return []
    points = build_comparison_points(
        f"2P-{burst_name}", base_rows, pred_rows,
        exact_fields=["alpha"],
        control_var_fields=["alpha", "beta"],
        nearest_field="beta", log_scale=True,
    )
    for pt in points:
        pt.burst_name = burst_name
    return points


def load_experiment_3p(
    results_dir: str, n_target: int, gamma: float, sweep_var: str,
) -> List[ComparisonPoint]:
    """実験 3 vs 3-P の比較点を生成 (delta / sigma スイープ別)."""
    base_rows = load_csv(os.path.join(results_dir, f"experiment_3_{sweep_var}.csv"))
    pred_rows = load_csv(os.path.join(
        results_dir, f"experiment_3P_{sweep_var}_nt{n_target}_g{gamma}.csv"
    ))
    log_scale = sweep_var == "sigma"
    return build_comparison_points(
        f"3P-{sweep_var}", base_rows, pred_rows,
        exact_fields=["alpha"],
        control_var_fields=["alpha", sweep_var],
        nearest_field=sweep_var, log_scale=log_scale,
    )


def load_experiment_4p(
    results_dir: str, n_target: int, gamma: float, burst_name: str,
) -> List[ComparisonPoint]:
    """実験 4 vs 4-P の比較点を生成 (バースト別, K 厳密一致 x rho 線形最近傍)."""
    base_rows = load_csv(os.path.join(results_dir, f"experiment_4_{burst_name}.csv"))
    pred_rows = load_csv(os.path.join(
        results_dir, f"experiment_4P_{burst_name}_nt{n_target}_g{gamma}.csv"
    ))
    if not base_rows:
        return []
    points = build_comparison_points(
        f"4P-{burst_name}", base_rows, pred_rows,
        exact_fields=["K"],
        control_var_fields=["K", "rho"],
        nearest_field="rho", log_scale=False,
    )
    for pt in points:
        pt.burst_name = burst_name
    return points


def format_pct(v: float) -> str:
    """改善率を Markdown 強調付き文字列に変換."""
    if v > 20:
        return f"**{v:+.1f}%**"
    elif v < -20:
        return f"_{v:+.1f}%_"
    else:
        return f"{v:+.1f}%"


def summarize_overall(points: List[ComparisonPoint]) -> None:
    """全体サマリを Markdown で出力."""
    print("## 全体サマリ\n")

    total = len(points)
    wins = sum(1 for pt in points if pt.is_predictive_better)
    much_wins = sum(1 for pt in points if pt.is_predictive_much_better)
    much_losses = sum(1 for pt in points if pt.is_predictive_much_worse)

    print(f"- **総データ点数**: {total}")
    print(f"- **Predictive 勝率**: {wins}/{total} ({wins/max(total,1)*100:.1f}%)")
    print(f"- **大幅改善 (>20%)**: {much_wins}/{total} ({much_wins/max(total,1)*100:.1f}%)")
    print(f"- **大幅悪化 (<-20%)**: {much_losses}/{total} ({much_losses/max(total,1)*100:.1f}%)")

    print("\n### 指標別勝率\n")
    print("| 指標 | 勝率 | 平均改善率 | 中央値改善率 |")
    print("|---|---|---|---|")
    for metric in METRICS:
        metric_pts = [pt for pt in points if pt.metric == metric]
        if not metric_pts:
            continue
        wins_m = sum(1 for pt in metric_pts if pt.is_predictive_better)
        improvements = sorted(pt.improvement_pct for pt in metric_pts)
        avg = sum(improvements) / len(improvements)
        median = improvements[len(improvements) // 2]
        print(f"| {metric} | {wins_m}/{len(metric_pts)} ({wins_m/len(metric_pts)*100:.1f}%) "
              f"| {avg:+.2f}% | {median:+.2f}% |")

    print("\n### 実験別勝率 (ERP 基準)\n")
    print("| 実験 | 勝率 (ERP) | 平均改善率 (ERP) | 大幅改善 | 大幅悪化 |")
    print("|---|---|---|---|---|")
    experiments = sorted(set(pt.experiment for pt in points))
    for exp in experiments:
        erp_pts = [pt for pt in points if pt.experiment == exp and pt.metric == "ERP"]
        if not erp_pts:
            continue
        wins_e = sum(1 for pt in erp_pts if pt.is_predictive_better)
        much_wins_e = sum(1 for pt in erp_pts if pt.is_predictive_much_better)
        much_losses_e = sum(1 for pt in erp_pts if pt.is_predictive_much_worse)
        avg = sum(pt.improvement_pct for pt in erp_pts) / len(erp_pts)
        print(f"| {exp} | {wins_e}/{len(erp_pts)} ({wins_e/len(erp_pts)*100:.1f}%) "
              f"| {avg:+.2f}% | {much_wins_e} | {much_losses_e} |")


def _pick_representative(values: List[float], n: int) -> List[float]:
    if len(values) <= n:
        return values
    step = len(values) / n
    idxs = sorted(set(min(len(values) - 1, int(round(i * step))) for i in range(n)))
    return [values[i] for i in idxs]


def summarize_by_condition(points: List[ComparisonPoint]) -> None:
    """バースト水準 x alpha の条件別サマリを Markdown で出力."""
    print("\n## 条件別サマリ (バースト x alpha)\n")

    # 実験 1-P (rho スイープ, 3 バースト水準)
    print("### 実験 1-P: バースト水準ごとの ERP 改善率 (rho 別)\n")
    exp1_pts = [pt for pt in points if pt.experiment == "1P" and pt.metric == "ERP"]
    if exp1_pts:
        bursts = sorted(set(pt.burst_name for pt in exp1_pts if pt.burst_name))
        rhos = sorted(set(pt.control_vars["rho"] for pt in exp1_pts))
        rhos_display = _pick_representative(rhos, 6)
        header = "| burst | " + " | ".join(f"rho={r:.2f}" for r in rhos_display) + " |"
        print(header)
        print("|" + "---|" * (len(rhos_display) + 1))
        for burst in bursts:
            row = f"| {burst} |"
            for rho in rhos_display:
                pt = next(
                    (p for p in exp1_pts
                     if p.burst_name == burst and abs(p.control_vars["rho"] - rho) < 1e-6),
                    None,
                )
                row += f" {format_pct(pt.improvement_pct) if pt else 'N/A'} |"
            print(row)
    else:
        print("_データなし_")

    # 実験 2-P (バースト別 x alpha x beta)
    for burst_name in ["medium", "strong"]:
        exp2_pts = [pt for pt in points if pt.experiment == f"2P-{burst_name}" and pt.metric == "ERP"]
        if not exp2_pts:
            continue
        print(f"\n### 実験 2-P ({burst_name} burst): alpha x beta の ERP 改善率\n")
        alphas = sorted(set(pt.control_vars["alpha"] for pt in exp2_pts))
        betas = sorted(set(pt.control_vars["beta"] for pt in exp2_pts))
        betas_display = _pick_representative(betas, 6)
        header = "| alpha \\ beta | " + " | ".join(f"{b:.3g}" for b in betas_display) + " |"
        print(header)
        print("|" + "---|" * (len(betas_display) + 1))
        for alpha in alphas:
            row = f"| {alpha:g} |"
            for beta in betas_display:
                pt = next(
                    (p for p in exp2_pts
                     if abs(p.control_vars["alpha"] - alpha) < 1e-6
                     and abs(p.control_vars["beta"] - beta) < 1e-6),
                    None,
                )
                row += f" {format_pct(pt.improvement_pct) if pt else 'N/A'} |"
            print(row)

    # 実験 3-P (delta / sigma sweep)
    for sweep_var in ["delta", "sigma"]:
        exp3_pts = [pt for pt in points if pt.experiment == f"3P-{sweep_var}" and pt.metric == "ERP"]
        if not exp3_pts:
            continue
        print(f"\n### 実験 3-P ({sweep_var} sweep): alpha x {sweep_var} の ERP 改善率\n")
        alphas = sorted(set(pt.control_vars["alpha"] for pt in exp3_pts))
        sweep_values = sorted(set(pt.control_vars[sweep_var] for pt in exp3_pts))
        sweep_display = _pick_representative(sweep_values, 6)
        header = f"| alpha \\ {sweep_var} | " + " | ".join(f"{s:.3g}" for s in sweep_display) + " |"
        print(header)
        print("|" + "---|" * (len(sweep_display) + 1))
        for alpha in alphas:
            row = f"| {alpha:g} |"
            for val in sweep_display:
                pt = next(
                    (p for p in exp3_pts
                     if abs(p.control_vars["alpha"] - alpha) < 1e-6
                     and abs(p.control_vars[sweep_var] - val) < 1e-6),
                    None,
                )
                row += f" {format_pct(pt.improvement_pct) if pt else 'N/A'} |"
            print(row)

    # 実験 4-P (バースト別 x K x rho)
    for burst_name in ["weak", "medium", "strong"]:
        exp4_pts = [pt for pt in points if pt.experiment == f"4P-{burst_name}" and pt.metric == "ERP"]
        if not exp4_pts:
            continue
        print(f"\n### 実験 4-P ({burst_name} burst): K x rho の ERP 改善率\n")
        Ks = sorted(set(int(pt.control_vars["K"]) for pt in exp4_pts))
        rhos = sorted(set(pt.control_vars["rho"] for pt in exp4_pts))
        rhos_display = _pick_representative(rhos, 5)
        header = "| K \\ rho | " + " | ".join(f"{r:.2f}" for r in rhos_display) + " |"
        print(header)
        print("|" + "---|" * (len(rhos_display) + 1))
        for K in Ks:
            row = f"| {K} |"
            for rho in rhos_display:
                pt = next(
                    (p for p in exp4_pts
                     if int(p.control_vars["K"]) == K and abs(p.control_vars["rho"] - rho) < 1e-6),
                    None,
                )
                row += f" {format_pct(pt.improvement_pct) if pt else 'N/A'} |"
            print(row)


def _fmt_num(v: float) -> str:
    if float(v).is_integer() and abs(v) < 1e6:
        return f"{int(v)}"
    return f"{v:.3g}"


def summarize_patterns(points: List[ComparisonPoint]) -> None:
    """パターン抽出 (Predictive が強い / 弱い条件)."""
    print("\n## パターン抽出\n")

    erp_pts = [pt for pt in points if pt.metric == "ERP"]
    if not erp_pts:
        print("_データなし_")
        return

    print("### 最も Predictive が有利な 10 条件 (ERP)\n")
    top_wins = sorted(erp_pts, key=lambda p: -p.improvement_pct)[:10]
    print("| 順位 | 実験 | burst | 制御変数 | ERP 改善率 |")
    print("|---|---|---|---|---|")
    for i, pt in enumerate(top_wins, 1):
        cv_str = ", ".join(f"{k}={_fmt_num(v)}" for k, v in pt.control_vars.items() if isinstance(v, float))
        print(f"| {i} | {pt.experiment} | {pt.burst_name or '-'} | {cv_str} | {format_pct(pt.improvement_pct)} |")

    print("\n### 最も Predictive が不利な 10 条件 (ERP)\n")
    top_losses = sorted(erp_pts, key=lambda p: p.improvement_pct)[:10]
    print("| 順位 | 実験 | burst | 制御変数 | ERP 改善率 |")
    print("|---|---|---|---|---|")
    for i, pt in enumerate(top_losses, 1):
        cv_str = ", ".join(f"{k}={_fmt_num(v)}" for k, v in pt.control_vars.items() if isinstance(v, float))
        print(f"| {i} | {pt.experiment} | {pt.burst_name or '-'} | {cv_str} | {format_pct(pt.improvement_pct)} |")

    print("\n### alpha 別の平均 ERP 改善率 (実験 2-P, 3-P)\n")
    print("| alpha | 2-P (medium) | 2-P (strong) | 3-P (delta) | 3-P (sigma) |")
    print("|---|---|---|---|---|")
    for alpha in [0.1, 1.0, 10.0]:
        row = f"| {alpha:g} |"
        for exp_tag in ["2P-medium", "2P-strong", "3P-delta", "3P-sigma"]:
            exp_pts = [
                pt for pt in erp_pts
                if pt.experiment == exp_tag
                and abs(pt.control_vars.get("alpha", -999) - alpha) < 1e-6
            ]
            if exp_pts:
                avg = sum(pt.improvement_pct for pt in exp_pts) / len(exp_pts)
                row += f" {avg:+.1f}% |"
            else:
                row += " N/A |"
        print(row)


def summarize_match_quality(points: List[ComparisonPoint]) -> None:
    """最近傍マッチングの品質に関する注記."""
    print("\n## データ点対応に関する注記\n")
    print(
        "ベース実験 (1〜4) は粗いグリッド (連続変数 3〜5 点), Predictive 実験 "
        "(1-P〜4-P) は密なグリッド (15〜40 点) で実行されているため, alpha / K / "
        "burst_name 等の離散変数は厳密一致, rho / beta / delta / sigma 等の連続変数は "
        "最近傍点マッチング (beta, sigma は対数距離, rho, delta は線形距離) で対応付けた."
    )
    by_exp: Dict[str, List[float]] = {}
    for pt in points:
        if pt.metric != "ERP":
            continue
        by_exp.setdefault(pt.experiment, []).append(pt.match_error)
    if by_exp:
        print("\n| 実験 | 最大マッチング誤差 | 平均マッチング誤差 |")
        print("|---|---|---|")
        for exp in sorted(by_exp):
            errs = by_exp[exp]
            print(f"| {exp} | {max(errs):.3f} | {sum(errs)/len(errs):.3f} |")
        print(
            "\n(誤差は対数距離を使う変数 (beta, sigma) では相対誤差, "
            "線形距離を使う変数 (rho, delta) では絶対誤差)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="CSV ディレクトリ")
    parser.add_argument("--n-target", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    points: List[ComparisonPoint] = []
    points += load_experiment_1p(args.results_dir, args.n_target, args.gamma)
    for burst in ["medium", "strong"]:
        points += load_experiment_2p(args.results_dir, args.n_target, args.gamma, burst)
    for sweep in ["delta", "sigma"]:
        points += load_experiment_3p(args.results_dir, args.n_target, args.gamma, sweep)
    for burst in ["weak", "medium", "strong"]:
        points += load_experiment_4p(args.results_dir, args.n_target, args.gamma, burst)

    print("# 実験 1-P〜4-P vs 実験 1〜4 の体系的評価\n")
    print(f"パラメータ: n_target={args.n_target}, gamma={args.gamma}\n")
    print("**凡例**:")
    print("- `+X.X%`: Predictive による改善率 (正=Predictive が良い, 負=Predictive が悪い)")
    print("- `**+XX%**`: 大幅改善 (>20%)")
    print("- `_-XX%_`: 大幅悪化 (<-20%)")
    print()

    if not points:
        print("**エラー**: 比較点が見つかりません. CSV ファイルの存在を確認してください.")
        return

    summarize_overall(points)
    summarize_by_condition(points)
    summarize_patterns(points)
    summarize_match_quality(points)

    print("\n---\n")
    print(f"総比較点数: {len(points)}")
    print("この Markdown を Claude チャットに貼り付けて分析を依頼できます.")


if __name__ == "__main__":
    main()
