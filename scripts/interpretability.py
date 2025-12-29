from src.interpret.cross_pollutant_attribution import cross_pollutant_importance
from src.interpret.feature_importance import permutation_importance
from src.interpret.temporal_attribution import compute_saliency_gru, summarize_saliency
from src.training.setup import get_dataloaders
from src.utils.cfg import load_config
from src.utils.device import get_device
from src.utils.helper import load_torch_model_from_registry


def main() -> None:
    """Run a combined interpretability analysis for multiple registered models.

    This script:
      - loads feature/target metadata from config
      - creates a test DataLoader (multi-target)
      - iterates over a list of registered model names and runs:
          1) permutation feature importance (ΔRMSE on NO2)
          2) temporal saliency (input-gradient magnitude)
          3) cross-pollutant attribution (input-gradients of NO2 w.r.t. pollutant features)

    The script prints results to stdout and continues to the next model if any
    individual analysis step fails.

    Returns:
        None.
    """
    cfg = load_config("config_training.yaml")
    device = get_device(cfg["training"].get("device", "auto"))

    model_names = ["multigru_best_model", "hgru_best_model", "gru_best_model"]

    data_cfg = cfg["data"]
    feature_cols = data_cfg["feature_cols"]
    target_cols = data_cfg["target_cols"]

    _, _, test_loader = get_dataloaders(cfg, multi_target=True)

    print(f"Feature columns: {feature_cols}")
    print(f"Target columns: {target_cols}")

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"   FULL ANALYSIS FOR MODEL: {name}")
        print(f"{'='*60}")

        model = load_torch_model_from_registry(name, device)

        print(f"\n[1] Permutation Importance (Target: NO2)")
        print("-" * 40)
        try:
            baseline_rmse, importances = permutation_importance(
                model=model,
                loader=test_loader,
                device=device,
                feature_names=feature_cols,
                target_cols=target_cols,
                target_of_interest="nitrogen_dioxide",
                n_repeats=5,
            )

            print(f"  Baseline RMSE (NO2): {baseline_rmse:.4f}")
            print("  Importance (ΔRMSE, sorted):")
            for feat, imp in sorted(
                importances.items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"    {feat:30s}  {imp:+.5f}")
        except Exception as e:
            print(f"  Failed: {e}")

        print(f"\n[2] Temporal Saliency (Gradient Magnitude)")
        print("-" * 40)
        try:
            out = compute_saliency_gru(
                model=model,
                loader=test_loader,
                device=device,
                target_cols=target_cols,
                target_of_interest="nitrogen_dioxide",
                max_batches=10,
            )
            ft_imp = out["feature_time_importance"]  # [L, F]

            per_feature = summarize_saliency(ft_imp, feature_cols)
            print("  Avg Saliency per Feature:")
            for feat_name, val in sorted(
                per_feature.items(), key=lambda kv: kv[1], reverse=True
            ):
                print(f"    {feat_name:30s}  {val:.6f}")

            per_time = ft_imp.mean(dim=1)  # [L]
            print("\n  Avg Saliency per Time Step (Last 5 steps):")
            L = len(per_time)
            for t in range(max(0, L - 5), L):
                print(f"    t={t:2d} (recent): {float(per_time[t]):.6f}")

        except Exception as e:
            print(f"  Failed: {e}")

        if "gru_best_model" in name and "multi" not in name and "hgru" not in name:
            print(f"\n[3] Cross-Pollutant Attribution")
            print("-" * 40)
            print(
                "  Skipping: Single-target GRU does not support cross-target attribution analysis."
            )
        else:
            print(f"\n[3] Cross-Pollutant Attribution")
            print("-" * 40)
            try:
                importance_cp = cross_pollutant_importance(
                    model=model,
                    loader=test_loader,
                    device=device,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    pollutants_of_interest=target_cols,
                    max_batches=10,
                )

                print("  Importance Score (Gradient Sum):")
                for p, val in sorted(
                    importance_cp.items(), key=lambda kv: kv[1], reverse=True
                ):
                    print(f"    {p:20s}  {val:.6f}")
            except Exception as e:
                print(f"  Failed: {e}")

        print("\n" + "_" * 60 + "\n")


if __name__ == "__main__":
    main()
