"""Run the complete signal detection system."""

from src.algo.example_workflow import complete_workflow_example

if __name__ == "__main__":
    # Run complete workflow
    data_path = "./data/AAPL_1_min.parquet"
    model_dir = "./models"

    print("=" * 80)
    print("RUNNING ML-BASED TRADING SIGNAL SYSTEM")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Model Directory: {model_dir}")
    print("=" * 80)

    signals, results, performance = complete_workflow_example(
        data_path=data_path,
        model_dir=model_dir,
        retrain=True,  # Train a new model
    )

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Generated {len(signals)} signals")
    print(f"Backtest results saved to {model_dir}/backtest_results.csv")
    print(f"Performance report saved to {model_dir}/performance_report.txt")
    print("=" * 80)

# Made with Bob
