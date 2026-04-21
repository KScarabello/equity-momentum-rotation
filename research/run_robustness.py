from __future__ import annotations

from research.robustness_grid import run_robustness_grid, print_robustness_report


def main() -> None:
    df = run_robustness_grid(
        stooq_dir="data_cache/stooq",
        base_config_path="config/alpha_v1.yaml",
        cost_bps=10.0,  # change to 0.0 to see gross performance
        ffill_limit=3,
        start=None,
        end=None,
    )
    print_robustness_report(df, top_k=15)

    # If you want the full table saved:
    # df.to_csv("robustness_grid_results.csv", index=False)


if __name__ == "__main__":
    main()
