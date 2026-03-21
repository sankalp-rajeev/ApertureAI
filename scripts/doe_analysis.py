"""
DOE Analysis: Analyze results from 27-configuration experiment.

Task 3.4 — Generates:
- Main effects plots (PSF, SNR, Vignette vs F1)
- Interaction heatmap (PSF x SNR)
- OLS regression for factor significance
- Resolution threshold (PSF where F1 < 0.90)

Usage:
    python scripts/doe_analysis.py --results results/doe_results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_results(csv_path: str) -> pd.DataFrame:
    """Load DOE results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def compute_main_effects(df: pd.DataFrame) -> dict:
    """Compute main effects for each factor."""
    # Group by each factor and compute mean F1
    psf_effects = df.groupby("psf_sigma")["macro_f1"].mean()
    snr_effects = df.groupby("snr_db")["macro_f1"].mean()
    vig_effects = df.groupby("vignette_strength")["macro_f1"].mean()

    return {
        "psf_sigma": psf_effects,
        "snr_db": snr_effects,
        "vignette_strength": vig_effects,
    }


def plot_main_effects(df: pd.DataFrame, output_dir: Path):
    """Create main effects plots for each factor."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # PSF effects
    psf_group = df.groupby("psf_sigma")["macro_f1"].agg(["mean", "std"]).reset_index()
    axes[0].errorbar(psf_group["psf_sigma"], psf_group["mean"],
                     yerr=psf_group["std"], marker="o", capsize=5, linewidth=2)
    axes[0].set_xlabel("PSF Sigma (pixels)")
    axes[0].set_ylabel("Mean Macro F1")
    axes[0].set_title("Effect of PSF Blur")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=0.90, color="r", linestyle="--", label="F1=0.90 threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SNR effects
    snr_group = df.groupby("snr_db")["macro_f1"].agg(["mean", "std"]).reset_index()
    axes[1].errorbar(snr_group["snr_db"], snr_group["mean"],
                     yerr=snr_group["std"], marker="s", capsize=5, linewidth=2, color="green")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Mean Macro F1")
    axes[1].set_title("Effect of Sensor Noise")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(y=0.90, color="r", linestyle="--", label="F1=0.90 threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Vignette effects
    vig_group = df.groupby("vignette_strength")["macro_f1"].agg(["mean", "std"]).reset_index()
    axes[2].errorbar(vig_group["vignette_strength"], vig_group["mean"],
                     yerr=vig_group["std"], marker="^", capsize=5, linewidth=2, color="orange")
    axes[2].set_xlabel("Vignette Strength")
    axes[2].set_ylabel("Mean Macro F1")
    axes[2].set_title("Effect of Illumination Non-uniformity")
    axes[2].set_ylim(0, 1.05)
    axes[2].axhline(y=0.90, color="r", linestyle="--", label="F1=0.90 threshold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "main_effects.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'main_effects.png'}")


def plot_interaction_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create PSF x SNR interaction heatmap."""
    # Pivot table: PSF vs SNR, mean F1
    pivot = df.pivot_table(
        values="macro_f1",
        index="psf_sigma",
        columns="snr_db",
        aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Macro F1"}
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSF Sigma (pixels)")
    plt.title("PSF x SNR Interaction Effect on Classification F1")
    plt.tight_layout()
    plt.savefig(output_dir / "interaction_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'interaction_heatmap.png'}")


def plot_class_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Plot per-class sensitivity to PSF blur."""
    plt.figure(figsize=(10, 6))

    for class_name in df["class"].unique():
        class_df = df[df["class"] == class_name]
        psf_group = class_df.groupby("psf_sigma")["f1_score"].mean()
        plt.plot(psf_group.index, psf_group.values, marker="o", label=class_name)

    plt.xlabel("PSF Sigma (pixels)")
    plt.ylabel("F1 Score")
    plt.title("Per-Class Sensitivity to PSF Blur")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.axhline(y=0.90, color="r", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "class_sensitivity.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'class_sensitivity.png'}")


def run_ols_analysis(df: pd.DataFrame) -> dict:
    """Run OLS regression to determine factor significance."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed. Skipping OLS analysis.")
        return None

    # Get unique configs (one row per config)
    config_df = df.drop_duplicates(subset=["config_id"])[
        ["psf_sigma", "snr_db", "vignette_strength", "macro_f1"]
    ].copy()

    # Normalize factors for comparable coefficients
    config_df["psf_norm"] = (config_df["psf_sigma"] - config_df["psf_sigma"].mean()) / config_df["psf_sigma"].std()
    config_df["snr_norm"] = (config_df["snr_db"] - config_df["snr_db"].mean()) / config_df["snr_db"].std()
    config_df["vig_norm"] = (config_df["vignette_strength"] - config_df["vignette_strength"].mean()) / (config_df["vignette_strength"].std() + 1e-8)

    # Fit OLS model
    model = smf.ols("macro_f1 ~ psf_norm + snr_norm + vig_norm", data=config_df).fit()

    print("\n" + "=" * 60)
    print("OLS Regression Results")
    print("=" * 60)
    print(model.summary())

    return {
        "r_squared": model.rsquared,
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "model": model,
    }


def find_resolution_threshold(df: pd.DataFrame) -> dict:
    """Find PSF sigma at which F1 drops below 0.90."""
    psf_effects = df.groupby("psf_sigma")["macro_f1"].mean().sort_index()

    threshold_psf = None
    for psf, f1 in psf_effects.items():
        if f1 < 0.90:
            threshold_psf = psf
            break

    # F1 degradation from best to worst config
    best_f1 = df["macro_f1"].max()
    worst_f1 = df["macro_f1"].min()
    degradation_pct = (best_f1 - worst_f1) * 100

    return {
        "threshold_psf": threshold_psf,
        "psf_effects": psf_effects.to_dict(),
        "best_f1": best_f1,
        "worst_f1": worst_f1,
        "degradation_pct": degradation_pct,
    }


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate summary report."""
    # Compute all analyses
    main_effects = compute_main_effects(df)
    threshold = find_resolution_threshold(df)
    ols = run_ols_analysis(df)

    # Find most sensitive class to PSF
    class_psf_sensitivity = {}
    for class_name in df["class"].unique():
        class_df = df[df["class"] == class_name]
        f1_at_psf_1 = class_df[class_df["psf_sigma"] == 1]["f1_score"].mean()
        f1_at_psf_6 = class_df[class_df["psf_sigma"] == 6]["f1_score"].mean()
        class_psf_sensitivity[class_name] = f1_at_psf_1 - f1_at_psf_6

    most_sensitive = max(class_psf_sensitivity, key=class_psf_sensitivity.get)

    # Write report
    report = f"""# DOE Analysis Report

## Summary

- **Total configurations:** 27 (3 factors x 3 levels)
- **Best macro F1:** {threshold['best_f1']:.4f}
- **Worst macro F1:** {threshold['worst_f1']:.4f}
- **F1 degradation:** {threshold['degradation_pct']:.1f}%

## Resolution Threshold

PSF sigma at which F1 drops below 0.90: **{threshold['threshold_psf']} pixels**

This corresponds to NA ~ 0.{10 if threshold['threshold_psf'] == 6 else 20 if threshold['threshold_psf'] == 3 else 50}

## Main Effects (Mean Macro F1)

### PSF Sigma
{pd.Series(main_effects['psf_sigma']).to_markdown()}

### SNR (dB)
{pd.Series(main_effects['snr_db']).to_markdown()}

### Vignette Strength
{pd.Series(main_effects['vignette_strength']).to_markdown()}

## Most Sensitive Defect Class

**{most_sensitive}** shows the highest sensitivity to PSF blur.
F1 drop from PSF=1 to PSF=6: {class_psf_sensitivity[most_sensitive]:.3f}

## Factor Significance (OLS)

| Factor | Coefficient | p-value |
|--------|-------------|---------|
"""
    if ols:
        for factor in ["psf_norm", "snr_norm", "vig_norm"]:
            coef = ols["coefficients"].get(factor, 0)
            pval = ols["p_values"].get(factor, 1)
            sig = "*" if pval < 0.05 else ""
            report += f"| {factor.replace('_norm', '')} | {coef:.4f} | {pval:.4f} {sig} |\n"

        dominant = max(
            ["psf_norm", "snr_norm", "vig_norm"],
            key=lambda x: abs(ols["coefficients"].get(x, 0))
        )
        report += f"\n**Dominant factor:** {dominant.replace('_norm', '')} (largest coefficient magnitude)\n"

    report += """
## Resume Bullet B2 (Template)

> Designed an **optical degradation simulator** modeling PSF convolution (Airy disk, NA 0.1-0.5),
> Poisson + Gaussian sensor noise, and illumination vignetting; ran a **27-configuration DOE**
> (3-factor x 3-level), identifying PSF sigma={}px as the resolution threshold beyond which F1
> degrades by **{:.1f}%**, informing camera/lens selection trade-offs for high-volume inspection.
""".format(threshold['threshold_psf'], threshold['degradation_pct'])

    report_path = output_dir / "doe_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze DOE results")
    parser.add_argument(
        "--results",
        type=str,
        default="results/doe_results.csv",
        help="Path to DOE results CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for plots and report"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    df = load_results(args.results)

    # Generate plots
    print("\nGenerating plots...")
    plot_main_effects(df, output_dir)
    plot_interaction_heatmap(df, output_dir)
    plot_class_sensitivity(df, output_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(df, output_dir)

    print("\n" + "=" * 60)
    print("DOE Analysis Complete!")
    print("=" * 60)
    print(f"  Main effects plot: {output_dir}/main_effects.png")
    print(f"  Interaction heatmap: {output_dir}/interaction_heatmap.png")
    print(f"  Class sensitivity: {output_dir}/class_sensitivity.png")
    print(f"  Report: {output_dir}/doe_report.md")


if __name__ == "__main__":
    main()
