from src.train import run_training_pipeline

if __name__ == "__main__":
    benchmark_df, best_model_name = run_training_pipeline()
    print("\n=== Model Benchmark ===")
    print(benchmark_df)
    print(f"\nBest model: {best_model_name}")