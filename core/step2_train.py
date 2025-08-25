import pandas as pd
from sklearn.preprocessing import StandardScaler
from convert import preprocess_creditcard_data
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib
import os

# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 读取数据
print("正在读取训练数据...")
train_df = pd.read_csv("./datasets/fraudTrain.csv", index_col='Unnamed: 0')
print("训练数据形状:", train_df.shape)

print("正在读取测试数据...")
test_df = pd.read_csv("./datasets/fraudTest.csv", index_col='Unnamed: 0')
print("测试数据形状:", test_df.shape)

# 预处理
X_train, y_train = preprocess_creditcard_data(train_df)
X_test, y_test = preprocess_creditcard_data(test_df)


# 显示部分预处理结果
print("\n训练特征 (X_train):")
print(X_train.head())

print("\n测试特征 (X_test):")
print(X_test.head())

# 使用 SMOTE 进行过采样
print("\n正在对训练数据应用 SMOTE 过采样...")
smote = SMOTE( random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("过采样后训练集类别分布:")
print(pd.Series(y_train_resampled).value_counts())



# 模型训练
print("\n正在训练随机森林分类器...")
clf = RandomForestClassifier(
    n_estimators=120,         # 增加树的数量（默认是100）
    min_samples_leaf=1,       #决策树的叶子节点（即最终的决策节点）上至少需要包含的样本数量
    n_jobs=-1,                # 使用所有CPU核心加速训练
    random_state=42,          # 保证结果可复现
    verbose=2                 # 显示训练进度（可选）
)

clf.fit(X_train_resampled, y_train_resampled)
# 测试集评估
print("\n正在使用测试集评估模型...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 保存模型
if not os.path.exists('./model'):
    os.makedirs('./model')
joblib.dump(clf, './model/random_forest_model.joblib')
print("模型已保存至 ./model/random_forest_model.joblib")
