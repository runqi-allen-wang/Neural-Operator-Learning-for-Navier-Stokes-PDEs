from pathlib import Path
import pandas as pd


TABLE_ROOT = Path("results/tables")
FINAL_DIR = TABLE_ROOT / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

SUITES = [
    "model_comparison",
    "data_scaling",
    "resolution_scaling",
    "depth_scaling",
]


def infer_suite(exp_name: str) -> str:
    exp_name = str(exp_name)
    for suite in SUITES:
        if exp_name.startswith(suite):
            return suite
    return "other"


def load_all_suite_csvs() -> pd.DataFrame:
    frames = []

    for suite in SUITES:
        suite_dir = TABLE_ROOT / suite
        if not suite_dir.exists():
            print(f"[WARN] missing directory: {suite_dir}")
            continue

        csv_files = sorted(suite_dir.glob("*.csv"))
        if not csv_files:
            print(f"[WARN] no csv files in: {suite_dir}")
            continue

        for path in csv_files:
            df = pd.read_csv(path)
            df["source_suite_dir"] = suite
            df["source_csv"] = str(path)
            frames.append(df)
            print(f"[OK] loaded {path}: {len(df)} rows")

    if not frames:
        raise RuntimeError("No CSV files found under results/tables/<suite>/")

    return pd.concat(frames, ignore_index=True)


def keep_latest_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "project.experiment_name" not in df.columns:
        raise ValueError("Missing column: project.experiment_name")

    if "run_name" in df.columns:
        df = df.sort_values(["project.experiment_name", "run_name"])
    elif "run_dir" in df.columns:
        df = df.sort_values(["project.experiment_name", "run_dir"])
    else:
        df["_row_id"] = range(len(df))
        df = df.sort_values(["project.experiment_name", "_row_id"])

    df = df.drop_duplicates("project.experiment_name", keep="last")

    if "_row_id" in df.columns:
        df = df.drop(columns=["_row_id"])

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "suite",
        "project.experiment_name",
        "run_name",
        "run_dir",
        "model.name",
        "data.path",
        "data.n_train",
        "data.n_val",
        "data.n_test",
        "data.train_resolution",
        "data.eval_resolution",
        "data.batch_size",
        "training.epochs",
        "model.depth",
        "model.width",
        "model.modes1",
        "model.modes2",
        "test_rel_l2",
        "test_mse",
        "test_mae",
        "test_rmse",
        "best_val_rel_l2",
        "best_epoch",
        "train_seconds",
        "n_parameters",
        "source_suite_dir",
        "source_csv",
    ]
    front = [c for c in preferred if c in df.columns]
    back = [c for c in df.columns if c not in front]
    return df[front + back]


def main():
    raw = load_all_suite_csvs()

    raw["suite"] = raw["project.experiment_name"].map(infer_suite)

    # Only keep the four target suites.
    raw = raw[raw["suite"].isin(SUITES)].copy()

    raw = reorder_columns(raw)
    raw.to_csv(FINAL_DIR / "summary_all_raw.csv", index=False)

    clean = keep_latest_per_experiment(raw)
    clean = reorder_columns(clean)
    clean.to_csv(FINAL_DIR / "summary_all_clean.csv", index=False)

    for suite in SUITES:
        sub = clean[clean["suite"] == suite].copy()
        out = FINAL_DIR / f"{suite}_clean.csv"
        sub.to_csv(out, index=False)
        print(f"[SAVE] {out}: {len(sub)} rows")

    print("\nFinal clean summary:")
    show_cols = [
        "suite",
        "project.experiment_name",
        "model.name",
        "data.n_train",
        "data.train_resolution",
        "model.depth",
        "test_rel_l2",
        "test_mse",
        "train_seconds",
        "n_parameters",
    ]
    show_cols = [c for c in show_cols if c in clean.columns]
    print(clean[show_cols].sort_values(["suite", "project.experiment_name"]).to_string(index=False))


if __name__ == "__main__":
    main()