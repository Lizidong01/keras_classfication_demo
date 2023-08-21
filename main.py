from data_process import read_data
from model import ClassficationModel
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_data('./data/Poem_classification - train_data.csv')
    model = ClassficationModel()
    model.model_fit(X_train, y_train)
    model.evaluate(X_test, y_test)

