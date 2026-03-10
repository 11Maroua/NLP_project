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

TARGET_COLUMN = "type"


def ensure_directories():
    """Create output directories if they do not exist."""
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
    return clean_text(text).split()


def build_full_text(df):
    """Create a combined text field for downstream analysis."""
    return (
        "titre: " + df["titre"].astype(str)
        + " ingredients: " + df["ingredients"].astype(str)
        + " recette: " + df["recette"].astype(str)
    )


def save_text_report(filename, content):
    """Save a text report to disk."""
    path = TEXT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def dataset_overview(df):
    """Generate and save the main dataset overview report."""
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


def print_length_statistics(df):
    """Print descriptive statistics for text lengths."""
    stats = df[["title_length", "ingredients_length", "recipe_length", "full_text_length"]].describe()
    print("\nTEXT LENGTH STATISTICS")
    print("=" * 80)
    print(stats.to_string())


def print_length_by_class(df):
    """Print text length statistics grouped by class."""
    print("\nTEXT LENGTH BY CLASS")
    print("=" * 80)

    for col in ["title_length", "ingredients_length", "recipe_length", "full_text_length"]:
        print(f"\n{col}")
        print(df.groupby(TARGET_COLUMN)[col].describe().round(2).to_string())


def plot_class_distribution(df):
    """Plot and save class distribution."""
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
    """Plot and save histogram for a text length feature."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=50)
    plt.title(title)
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_length_boxplot_by_class(df, column, title, filename):
    """Plot and save boxplot of text length by class."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y=column)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of words")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def print_top_words_per_class(df, text_column="recette", top_n=20, min_word_length=3):
    """Print most frequent words per class for a given text column."""
    print(f"\nTOP WORDS PER CLASS - COLUMN: {text_column}")
    print("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, text_column].astype(str)
        tokens = []
        for text in texts:
            tokens.extend([tok for tok in tokenize(text) if len(tok) >= min_word_length])

        counter = Counter(tokens)
        print(f"\nClass: {label}")
        for word, count in counter.most_common(top_n):
            print(f"{word}\t{count}")


def print_top_ingredients_per_class(df, top_n=20, min_word_length=3):
    """Print most frequent ingredient terms per class."""
    print("\nTOP INGREDIENT TERMS PER CLASS")
    print("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, "ingredients"].astype(str)
        tokens = []
        for text in texts:
            tokens.extend([tok for tok in tokenize(text) if len(tok) >= min_word_length])

        counter = Counter(tokens)
        print(f"\nClass: {label}")
        for word, count in counter.most_common(top_n):
            print(f"{word}\t{count}")


def print_top_tfidf_terms_global(df, text_column="full_text", top_n=30, max_features=5000):
    """Print top global TF-IDF terms."""
    texts = df[text_column].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = X.mean(axis=0).A1
    top_indices = mean_scores.argsort()[::-1][:top_n]

    print(f"\nTOP GLOBAL TF-IDF TERMS - COLUMN: {text_column}")
    print("=" * 80)
    for idx in top_indices:
        print(f"{feature_names[idx]}\t{mean_scores[idx]:.6f}")


def print_top_tfidf_terms_per_class(df, text_column="full_text", top_n=20, max_features=5000):
    """Print top TF-IDF terms for each class."""
    print(f"\nTOP TF-IDF TERMS PER CLASS - COLUMN: {text_column}")
    print("=" * 80)

    for label in sorted(df[TARGET_COLUMN].unique()):
        texts = df.loc[df[TARGET_COLUMN] == label, text_column].astype(str).apply(clean_text)

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = X.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:top_n]

        print(f"\nClass: {label}")
        for idx in top_indices:
            print(f"{feature_names[idx]}\t{mean_scores[idx]:.6f}")


def print_most_common_title_patterns(df, top_n=20):
    """Print most common title unigrams and bigrams."""
    titles = df["titre"].astype(str).apply(clean_text)

    unigram_counter = Counter()
    bigram_counter = Counter()

    for title in titles:
        tokens = [tok for tok in title.split() if len(tok) >= 2]
        unigram_counter.update(tokens)
        bigrams = zip(tokens, tokens[1:])
        bigram_counter.update([" ".join(bg) for bg in bigrams])

    print("\nMOST COMMON TITLE PATTERNS")
    print("=" * 80)

    print("\nTop title unigrams:")
    for term, count in unigram_counter.most_common(top_n):
        print(f"{term}\t{count}")

    print("\nTop title bigrams:")
    for term, count in bigram_counter.most_common(top_n):
        print(f"{term}\t{count}")


def print_outlier_examples(df, n=5):
    """Print a few longest recipes for qualitative inspection."""
    longest = df.sort_values("recipe_length", ascending=False).head(n)

    print("\nLONGEST RECIPE EXAMPLES")
    print("=" * 80)

    for i, (_, row) in enumerate(longest.iterrows(), start=1):
        print(f"\nExample {i}")
        print(f"doc_id: {row['doc_id']}")
        print(f"type: {row['type']}")
        print(f"title: {row['titre']}")
        print(f"recipe_length: {row['recipe_length']}")
        print(f"ingredients_length: {row['ingredients_length']}")
        preview = str(row["recette"])[:600].replace("\n", " ")
        print(f"recipe_preview: {preview}")


def main():
    sns.set_theme(style="whitegrid")
    ensure_directories()

    df = load_data(DATA_PATH)
    df = add_length_features(df)

    dataset_overview(df)

    print_length_statistics(df)
    print_length_by_class(df)

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

    print_top_words_per_class(df, text_column="titre", top_n=20)
    print_top_words_per_class(df, text_column="ingredients", top_n=20)
    print_top_words_per_class(df, text_column="recette", top_n=20)

    print_top_ingredients_per_class(df, top_n=25)

    print_top_tfidf_terms_global(df, text_column="full_text", top_n=30)
    print_top_tfidf_terms_per_class(df, text_column="full_text", top_n=20)

    print_most_common_title_patterns(df, top_n=20)
    print_outlier_examples(df, n=5)

    print("\nEDA completed.")
    print(f"Report saved to: {TEXT_DIR / 'dataset_overview.txt'}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()