import pandas as pd

def remove_is_fraud_column():
    file_paths = ["./datasets/fraudTrain.csv", "./datasets/fraudTest.csv"]
    for file_path in file_paths:
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 检查是否存在 is_fraud 列
            if 'is_fraud' in df.columns:
                # 去掉 is_fraud 列
                df = df.drop(columns=['is_fraud'])

            # 生成新的文件名
            new_file_path = file_path.rsplit('.', 1)[0] + '_new.csv'

            # 保存新的 CSV 文件
            df.to_csv(new_file_path, index=False)
            print(f"已处理 {file_path}，新文件保存为 {new_file_path}")
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到，请检查文件路径。")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")

if __name__ == "__main__":
    remove_is_fraud_column()
