import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def read_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Poem'])
    minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = minilm_model.encode(data['Poem'].tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, data['Genre'], test_size=0.2, random_state=42
    )
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, X_test, y_train, y_test
