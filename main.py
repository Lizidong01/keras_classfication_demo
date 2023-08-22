from data_process import read_data
from model import ClassificationModel
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_data('./data/Poem_classification - train_data.csv')
    model = ClassificationModel(384)
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.predict(X_test)

