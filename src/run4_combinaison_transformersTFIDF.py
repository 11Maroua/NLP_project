# -*- coding: utf-8 -*-
import gc
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

from utils import load_data, preprocess, build_text


RANDOM_STATE = 42
TEXT_COLUMNS = ["titre", "ingredients", "recette"]

RESULTS_DIR = "results"
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
CAMEMBERT_DIM = 768


def ensure_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def prepare_texts(train_df, test_df, columns):
    x_train = build_text(train_df, columns).astype(str).apply(preprocess)
    x_test = build_text(test_df, columns).astype(str).apply(preprocess)
    return x_train, x_test


def build_tfidf_features(x_train_text, x_test_text):
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train_text)
    x_test_tfidf = vectorizer.transform(x_test_text)
    return x_train_tfidf, x_test_tfidf


def train_word2vec(x_train_text):
    sentences = [simple_preprocess(text) for text in x_train_text]

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        epochs=10,
        seed=RANDOM_STATE,
    )
    return model


def document_vector_word2vec(text, model):
    tokens = simple_preprocess(str(text))
    vectors = [model.wv[token] for token in tokens if token in model.wv]

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def build_word2vec_features(x_train_text, x_test_text):
    print("Training Word2Vec...")
    w2v_model = train_word2vec(x_train_text)

    print("Building Word2Vec train vectors...")
    x_train_w2v = np.array(
        [document_vector_word2vec(text, w2v_model) for text in x_train_text],
        dtype=np.float32,
    )

    print("Building Word2Vec test vectors...")
    x_test_w2v = np.array(
        [document_vector_word2vec(text, w2v_model) for text in x_test_text],
        dtype=np.float32,
    )

    return x_train_w2v, x_test_w2v


def load_camembert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device for CamemBERT:", device)

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModel.from_pretrained("camembert-base")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def build_camembert_cache_path(split_name, n_rows, max_length):
    filename = "camembert_{}_n{}_len{}.npy".format(split_name, n_rows, max_length)
    return os.path.join(CACHE_DIR, filename)


def encode_texts_to_npy_memmap(
    texts,
    split_name,
    tokenizer,
    model,
    device,
    max_length=32,
    batch_size=2,
):
    """
    Encode texts with CamemBERT and write embeddings directly into a .npy memmap.
    This avoids the final in-memory copy that was causing the bus error.
    """
    texts = list(texts)
    n_rows = len(texts)
    cache_path = build_camembert_cache_path(split_name, n_rows, max_length)

    if os.path.exists(cache_path):
        print("Loading cached CamemBERT embeddings from:", cache_path)
        return np.load(cache_path, mmap_mode="r")

    print("Creating CamemBERT cache:", cache_path)

    embeddings_memmap = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, CAMEMBERT_DIM),
    )

    write_idx = 0

    for start_idx in range(0, n_rows, batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = (
            outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float16)
        )

        current_batch_size = batch_embeddings.shape[0]
        embeddings_memmap[write_idx:write_idx + current_batch_size] = batch_embeddings
        write_idx += current_batch_size

        processed = min(start_idx + batch_size, n_rows)
        print("CamemBERT {}: {}/{}".format(split_name, processed, n_rows))

        del inputs, outputs, batch_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings_memmap.flush()
    del embeddings_memmap
    gc.collect()

    return np.load(cache_path, mmap_mode="r")


def build_camembert_features(x_train_text, x_test_text, max_length=32, batch_size=2):
    print("Loading CamemBERT...")
    tokenizer, model, device = load_camembert()

    print("Building CamemBERT train embeddings...")
    x_train_camembert = encode_texts_to_npy_memmap(
        texts=x_train_text,
        split_name="train",
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    print("Building CamemBERT test embeddings...")
    x_test_camembert = encode_texts_to_npy_memmap(
        texts=x_test_text,
        split_name="test",
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    del tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return x_train_camembert, x_test_camembert


def combine_sparse_and_dense(x_sparse, x_dense):
    x_dense = np.asarray(x_dense, dtype=np.float32)
    x_dense_sparse = csr_matrix(x_dense)
    return hstack([x_sparse, x_dense_sparse], format="csr")


def evaluate_model(model_name, x_train, x_test, y_train, y_test):
    clf = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print("Macro-F1: {:.4f}".format(f1))

    return f1, y_pred


def save_results(y_test, pred_w2v, pred_camembert, f1_w2v, f1_camembert):
    results_df = pd.DataFrame(
        {
            "model": [
                "TF-IDF + Word2Vec + SVM",
                "TF-IDF + CamemBERT + SVM",
            ],
            "macro_f1": [
                round(f1_w2v, 4),
                round(f1_camembert, 4),
            ],
        }
    )
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "results_run4_comparison.csv"),
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "pred_tfidf_w2v_svm": pred_w2v,
            "pred_tfidf_camembert_svm": pred_camembert,
        }
    )
    predictions_df.to_csv(
        os.path.join(RESULTS_DIR, "predictions_run4_comparison.csv"),
        index=False,
    )

    print("\nSaved files:")
    print("- results/results_run4_comparison.csv")
    print("- results/predictions_run4_comparison.csv")


def main():
    ensure_directories()

    print("Loading data...")
    train, test = load_data()

    QUICK_TEST = False

    if QUICK_TEST:
        train = train.sample(n=min(1000, len(train)), random_state=RANDOM_STATE)
        test = test.sample(n=min(300, len(test)), random_state=RANDOM_STATE)
        print("Quick test mode enabled.")
        print("Train size:", len(train))
        print("Test size:", len(test))

    y_train = train["type"]
    y_test = test["type"]

    print("Preparing text...")
    x_train_text, x_test_text = prepare_texts(train, test, TEXT_COLUMNS)

    print("Building TF-IDF features...")
    x_train_tfidf, x_test_tfidf = build_tfidf_features(x_train_text, x_test_text)

    print("Building Word2Vec features...")
    x_train_w2v, x_test_w2v = build_word2vec_features(x_train_text, x_test_text)

    print("Combining TF-IDF + Word2Vec...")
    x_train_tfidf_w2v = combine_sparse_and_dense(x_train_tfidf, x_train_w2v)
    x_test_tfidf_w2v = combine_sparse_and_dense(x_test_tfidf, x_test_w2v)

    f1_tfidf_w2v, pred_tfidf_w2v = evaluate_model(
        model_name="TF-IDF + Word2Vec + SVM",
        x_train=x_train_tfidf_w2v,
        x_test=x_test_tfidf_w2v,
        y_train=y_train,
        y_test=y_test,
    )

    del x_train_w2v, x_test_w2v, x_train_tfidf_w2v, x_test_tfidf_w2v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building CamemBERT features...")
    x_train_camembert, x_test_camembert = build_camembert_features(
        x_train_text,
        x_test_text,
        max_length=32,
        batch_size=2,
    )

    print("Combining TF-IDF + CamemBERT...")
    x_train_tfidf_camembert = combine_sparse_and_dense(x_train_tfidf, x_train_camembert)
    x_test_tfidf_camembert = combine_sparse_and_dense(x_test_tfidf, x_test_camembert)

    del x_train_camembert, x_test_camembert
    gc.collect()

    f1_tfidf_camembert, pred_tfidf_camembert = evaluate_model(
        model_name="TF-IDF + CamemBERT + SVM",
        x_train=x_train_tfidf_camembert,
        x_test=x_test_tfidf_camembert,
        y_train=y_train,
        y_test=y_test,
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print("TF-IDF + Word2Vec + SVM  : {:.4f}".format(f1_tfidf_w2v))
    print("TF-IDF + CamemBERT + SVM : {:.4f}".format(f1_tfidf_camembert))

    if f1_tfidf_camembert > f1_tfidf_w2v:
        print("Best model: TF-IDF + CamemBERT + SVM")
    elif f1_tfidf_camembert < f1_tfidf_w2v:
        print("Best model: TF-IDF + Word2Vec + SVM")
    else:
        print("Best model: tie")

    save_results(
        y_test=y_test,
        pred_w2v=pred_tfidf_w2v,
        pred_camembert=pred_tfidf_camembert,
        f1_w2v=f1_tfidf_w2v,
        f1_camembert=f1_tfidf_camembert,
    )


if __name__ == "__main__":
    main()