from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "lib" / "StressLevel" / "StressLevelDataset.csv"
DISCRETIZED_DATA_PATH = ROOT_DIR / "results" / "StressLevel" / "StressLevel_Discretized.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "StressLevel"
TARGET_COLUMN = "anxiety_level"
TOP_K_FEATURES = 5
RANDOM_STATE = 42


def prepare_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.copy()

    if df[TARGET_COLUMN].nunique() > 3:
        df[TARGET_COLUMN] = pd.cut(df[TARGET_COLUMN], bins=3, labels=False, include_lowest=True)
        df["self_esteem"] = pd.cut(df["self_esteem"], bins=3, labels=False, include_lowest=True)
        df["depression"] = pd.cut(df["depression"], bins=3, labels=False, include_lowest=True)

    return df


def load_prepared_dataset() -> pd.DataFrame:
    if DISCRETIZED_DATA_PATH.exists():
        return pd.read_csv(DISCRETIZED_DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_dataset(DATA_PATH)
    df.to_csv(DISCRETIZED_DATA_PATH, index=False)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba")

    return X_train, X_test, y_train, y_test


def rank_features_by_mutual_information(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    mi_scores = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=RANDOM_STATE)
    ranking = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_information": mi_scores,
        }
    ).sort_values("mutual_information", ascending=False)
    return ranking


def evaluate_model(model, X_train, X_test, y_train, y_test, features, label, output_dir: Path):
    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)
    matrix = confusion_matrix(y_test, predictions)
    
    print(f"\n===== {label} =====")
    print(f"Características usadas: {len(features)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(matrix)

    result_text = [
        f"MODELO: {label}",
        f"Características: {', '.join(features)}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "Classification report:",
        report,
        "",
        "Confusion matrix:",
        str(matrix),
        "",
    ]

    report_path = output_dir / f"{label.replace(' ', '_').lower()}_report.txt"
    report_path.write_text("\n".join(result_text), encoding="utf-8")

    return {
        "model": label,
        "accuracy": accuracy,
        "report_path": str(report_path),
    }


def save_feature_ranking(ranking: pd.DataFrame, output_dir: Path):
    ranking_path = output_dir / "ranking_informacion_mutua.csv"
    ranking.to_csv(ranking_path, index=False)

    plt.figure(figsize=(10, max(4, 0.35 * len(ranking))))
    sns.barplot(data=ranking, x="mutual_information", y="feature", palette="viridis")
    plt.title("Ranking de características por información mutua")
    plt.xlabel("Información mutua")
    plt.ylabel("Característica")
    plt.tight_layout()
    plt.savefig(output_dir / "ranking_informacion_mutua.png", dpi=300)
    plt.close()

    return ranking_path


def save_decision_tree_plot(model, feature_names, output_dir: Path, filename: str):
    plt.figure(figsize=(24, 12))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=["0", "1", "2"], fontsize=12, max_depth=3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight", )
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_prepared_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    ranking = rank_features_by_mutual_information(X_train, y_train)
    save_feature_ranking(ranking, OUTPUT_DIR)

    top_k = min(TOP_K_FEATURES, len(ranking))
    top_features = ranking.head(top_k)["feature"].tolist()
    all_features = list(X_train.columns)

    print("\n=== Ranking de características ===")
    print(ranking.to_string(index=False))
    print(f"\nTop {top_k} características: {top_features}")

    summary = []

    summary.append(
        evaluate_model(
            DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
            X_train,
            X_test,
            y_train,
            y_test,
            all_features,
            "arbol_decision_todas_las_caracteristicas",
            OUTPUT_DIR,
        )
    )
    save_decision_tree_plot(
        DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE).fit(X_train[all_features], y_train),
        all_features,
        OUTPUT_DIR,
        "arbol_decision_todas_las_caracteristicas.png",
    )

    summary.append(
        evaluate_model(
            CategoricalNB(),
            X_train,
            X_test,
            y_train,
            y_test,
            all_features,
            "bayes_ingenuo_todas_las_caracteristicas",
            OUTPUT_DIR,
        )
    )

    summary.append(
        evaluate_model(
            DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
            X_train,
            X_test,
            y_train,
            y_test,
            top_features,
            f"arbol_decision_top_{top_k}_caracteristicas",
            OUTPUT_DIR,
        )
    )
    save_decision_tree_plot(
        DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE).fit(X_train[top_features], y_train),
        top_features,
        OUTPUT_DIR,
        f"arbol_decision_top_{top_k}_caracteristicas.png",
    )

    summary.append(
        evaluate_model(
            CategoricalNB(),
            X_train,
            X_test,
            y_train,
            y_test,
            top_features,
            f"bayes_ingenuo_top_{top_k}_caracteristicas",
            OUTPUT_DIR,
        )
    )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "resumen_modelos.csv", index=False)

    print("\n=== Resumen final ===")
    print(summary_df.to_string(index=False))
    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()