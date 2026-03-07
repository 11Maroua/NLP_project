# -*- coding: utf-8 -*-
from pathlib import Path
from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/train.csv")
OUTPUT_DIR = Path("results/eda")
FIGURES_DIR = OUTPUT_DIR / "figures"
TEXT_DIR = OUTPUT_DIR / "reports"

TEXT_COLUMNS = ["titre", "ingredients", "recette"]
TARGET_COLUMN = "type"


def ensure_directories():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def clean_text(text):
    """
    Basic French-friendly text cleaning.
    Keeps accented characters and removes punctuation/digits.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüçœæ\s-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    """Tokenize cleaned text into words."""
    text = clean_text(text)
    return text.split()


def build_full_text(df):
    """Create a combined text field for downstream analysis."""
    return (
        "titre: " + df["titre"].astype(str)
        + " ingredients: " + df["ingredients"].astype(str)
        + " recette: " + df["recette"].astype(str)
    )


def save_text_report(filename, content):
    path = TEXT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def dataset_overview(df):
    lines = []
    lines.append("DATASET OVERVIEW")
    lines.append("=" * 80)
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append("\nMissing values:")
    lines.append(df.isnull().sum().to_string())
    lines.append("\nClass distribution:")
    lines.append(df[TARGET_COLUMN].value_counts().to_string())
    lines.append("\nClass distribution (%):")
    lines.append((df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_string())

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report("dataset_overview.txt", report)


def add_length_features(df):
    """Add word count features for each text field."""
    df = df.copy()
    df["title_length"] = df["titre"].astype(str).apply(lambda x: len(tokenize(x)))
    df["ingredients_length"] = df["ingredients"].astype(str).apply(lambda x: len(tokenize(x)))
    df["recipe_length"] = df["recette"].astype(str).apply(lambda x: len(tokenize(x)))
    df["full_text"] = build_full_text(df)
    df["full_text_length"] = df["full_text"].astype(str).apply(lambda x: len(tokenize(x)))
    return df


def length_statistics(df):
    lines = []
    lines.append("TEXT LENGTH STATISTICS")
    lines.append("=" * 80)
    stats = df[["title_length", "ingredients_length", "recipe_length", "full_text_length"]].describe()
    lines.append(stats.to_string())

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report("length_statistics.txt", report)


def length_by_class(df):
    lines = []
    lines.append("TEXT LENGTH BY CLASS")
    lines.append("=" * 80)

    for col in ["title_length", "ingredients_length", "recipe_length", "full_text_length"]:
        lines.append(f"\n{col}")
        lines.append(df.groupby(TARGET_COLUMN)[col].describe().round(2).to_string())

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report("length_by_class.txt", report)


def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))
    order = df[TARGET_COLUMN].value_counts().index
    sns.countplot(data=df, x=TARGET_COLUMN, order=order)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()


def plot_text_length_histogram(df, column, title, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=50)
    plt.title(title)
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_length_boxplot_by_class(df, column, title, filename):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y=column)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of words")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def get_top_words_per_class(df, text_column="recette", top_n=20, min_word_length=3):
    lines = []
    lines.append(f"TOP WORDS PER CLASS - COLUMN: {text_column}")
    lines.append("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, text_column].astype(str)
        tokens = []
        for text in texts:
            tokens.extend([tok for tok in tokenize(text) if len(tok) >= min_word_length])

        counter = Counter(tokens)
        lines.append(f"\nClass: {label}")
        for word, count in counter.most_common(top_n):
            lines.append(f"{word}\t{count}")

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report(f"top_words_per_class_{text_column}.txt", report)


def get_top_ingredients_per_class(df, top_n=20, min_word_length=3):
    lines = []
    lines.append("TOP INGREDIENT TERMS PER CLASS")
    lines.append("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, "ingredients"].astype(str)
        tokens = []
        for text in texts:
            tokens.extend([tok for tok in tokenize(text) if len(tok) >= min_word_length])

        counter = Counter(tokens)
        lines.append(f"\nClass: {label}")
        for word, count in counter.most_common(top_n):
            lines.append(f"{word}\t{count}")

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report("top_ingredients_per_class.txt", report)


def top_tfidf_terms_global(df, text_column="full_text", top_n=30, max_features=5000):
    texts = df[text_column].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = X.mean(axis=0).A1
    top_indices = mean_scores.argsort()[::-1][:top_n]

    lines = []
    lines.append(f"TOP GLOBAL TF-IDF TERMS - COLUMN: {text_column}")
    lines.append("=" * 80)
    for idx in top_indices:
        lines.append(f"{feature_names[idx]}\t{mean_scores[idx]:.6f}")

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report(f"top_tfidf_global_{text_column}.txt", report)


def top_tfidf_terms_per_class(df, text_column="full_text", top_n=20, max_features=5000):
    lines = []
    lines.append(f"TOP TF-IDF TERMS PER CLASS - COLUMN: {text_column}")
    lines.append("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, text_column].astype(str).apply(clean_text)

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = X.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]

        lines.append(f"\nClass: {label}")
        for idx in top_indices:
            lines.append(f"{feature_names[idx]}\t{mean_scores[idx]:.6f}")

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report(f"top_tfidf_per_class_{text_column}.txt", report)


def most_common_title_patterns(df, top_n=20):
    """
    Extract common title unigrams and bigrams.
    Useful to see whether titles are strongly discriminative.
    """
    titles = df["titre"].astype(str).apply(clean_text)

    unigram_counter = Counter()
    bigram_counter = Counter()

    for title in titles:
        tokens = [tok for tok in title.split() if len(tok) >= 2]
        unigram_counter.update(tokens)
        bigrams = zip(tokens, tokens[1:])
        bigram_counter.update([" ".join(bg) for bg in bigrams])

    lines = []
    lines.append("MOST COMMON TITLE PATTERNS")
    lines.append("=" * 80)
    lines.append("\nTop title unigrams:")
    for term, count in unigram_counter.most_common(top_n):
        lines.append(f"{term}\t{count}")

    lines.append("\nTop title bigrams:")
    for term, count in bigram_counter.most_common(top_n):
        lines.append(f"{term}\t{count}")

    report = "\n".join(lines)
    print("\n" + report)
    save_text_report("title_patterns.txt", report)


def outlier_examples(df, n=5):
    """Save a few longest recipes for qualitative inspection."""
    longest = df.sort_values("recipe_length", ascending=False).head(n)

    lines = []
    lines.append("LONGEST RECIPE EXAMPLES")
    lines.append("=" * 80)

    for i, (_, row) in enumerate(longest.iterrows(), start=1):
        lines.append(f"\nExample {i}")
        lines.append(f"doc_id: {row['doc_id']}")
        lines.append(f"type: {row['type']}")
        lines.append(f"title: {row['titre']}")
        lines.append(f"recipe_length: {row['recipe_length']}")
        lines.append(f"ingredients_length: {row['ingredients_length']}")
        preview = str(row["recette"])[:600].replace("\n", " ")
        lines.append(f"recipe_preview: {preview}")

    report = "\n".join(lines)
    save_text_report("longest_recipe_examples.txt", report)


def main():
    sns.set_theme(style="whitegrid")
    ensure_directories()

    df = load_data(DATA_PATH)
    df = add_length_features(df)

    dataset_overview(df)
    length_statistics(df)
    length_by_class(df)

    plot_class_distribution(df)
    plot_text_length_histogram(
        df,
        column="recipe_length",
        title="Recipe Length Distribution",
        filename="recipe_length_distribution.png",
    )
    plot_text_length_histogram(
        df,
        column="ingredients_length",
        title="Ingredients Length Distribution",
        filename="ingredients_length_distribution.png",
    )
    plot_text_length_histogram(
        df,
        column="title_length",
        title="Title Length Distribution",
        filename="title_length_distribution.png",
    )

    plot_length_boxplot_by_class(
        df,
        column="recipe_length",
        title="Recipe Length by Class",
        filename="recipe_length_by_class_boxplot.png",
    )
    plot_length_boxplot_by_class(
        df,
        column="ingredients_length",
        title="Ingredients Length by Class",
        filename="ingredients_length_by_class_boxplot.png",
    )
    plot_length_boxplot_by_class(
        df,
        column="title_length",
        title="Title Length by Class",
        filename="title_length_by_class_boxplot.png",
    )

    get_top_words_per_class(df, text_column="titre", top_n=20)
    get_top_words_per_class(df, text_column="ingredients", top_n=20)
    get_top_words_per_class(df, text_column="recette", top_n=20)

    get_top_ingredients_per_class(df, top_n=25)

    top_tfidf_terms_global(df, text_column="full_text", top_n=30)
    top_tfidf_terms_per_class(df, text_column="full_text", top_n=20)

    most_common_title_patterns(df, top_n=20)
    outlier_examples(df, n=5)

    print("\nEDA completed.")
    print(f"Reports saved to: {TEXT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()