import pandas as pd
import re
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stemmer = SnowballStemmer('french')
stop_fr = set(stopwords.words('french'))

def load_data():
    train = pd.read_csv('data/train.csv')
    test  = pd.read_csv('data/test.csv')
    return train, test

def preprocess(text, stem=False):
    # 1. Minuscules
    text = text.lower()
    # 2. Supprimer caractères spéciaux et chiffres
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    # 3. Tokeniser
    tokens = text.split()
    # 4. Supprimer stopwords et mots trop courts
    tokens = [t for t in tokens if t not in stop_fr and len(t) > 2]
    # 5. Stemming 
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def build_text(df, colonnes=['titre', 'ingredients', 'recette']):
    """Concatène les colonnes texte choisies pour former un meme texte etappliquet tf-idf dessus directement"""
    return df[colonnes].fillna('').apply(lambda r: ' '.join(r), axis=1)