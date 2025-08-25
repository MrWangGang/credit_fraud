import pandas as pd
from convert import preprocess_creditcard_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def test_model(model_path, test_data_path):
    # 读取模型
    clf = joblib.load(model_path)

    # 读取测试数据
    print("正在读取测试数据...")
    test_df = pd.read_csv(test_data_path, index_col='Unnamed: 0')
    print("测试数据形状:", test_df.shape)

    # 数据预处理
    X_test, y_test = preprocess_creditcard_data(test_df)

    # 模型预测
    print("\n正在对测试集进行评估...")
    y_pred = clf.predict(X_test)

    # 评估结果输出
    print("\n分类报告:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    model_path = './model/random_forest_model.joblib'
    test_data_path = './datasets/fraudTest.csv'
    test_model(model_path, test_data_path)
