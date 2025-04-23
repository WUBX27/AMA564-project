import os
import pandas as pd
import fasttext
from sklearn.metrics import f1_score
import time

# Get the absolute path of the current script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Change the working directory to the directory where the current script is located
os.chdir(current_dir)

dataset_dir = os.path.join(current_dir, 'dataset')

datasets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# 将标签转换为 FastText 格式
def convert_to_fasttext_format(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = row['text']
            label = f"__label__{row['label']}"
            if text and label:  # 确保文本和标签都存在
                f.write(f"{label} {text}\n")

# 评估模型
def evaluate_model(model, test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    true_labels = []
    predictions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue  # 跳过格式不正确的行
        label, text = parts
        if not text:
            continue  # 跳过文本为空的行
        
        # 检查标签格式
        label_parts = label.split('__label__')
        if len(label_parts) != 2:
            continue  # 跳过标签格式不正确的行
        
        true_label = int(label_parts[1])
        predicted_label = int(model.predict(text)[0][0].split('__label__')[1])
        true_labels.append(true_label)
        predictions.append(predicted_label)
    
    return true_labels, predictions
# 记录结果
results = []

# 遍历每个数据集
for dataset in datasets:
    dataset_path = os.path.join(dataset_dir, dataset)
    train_file = os.path.join(dataset_path, 'train.csv')
    test_file = os.path.join(dataset_path, 'test.csv')

    # 读取训练集和测试集
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # 将训练集和测试集转换为 FastText 格式
    train_fasttext_file = os.path.join(dataset_path, 'train_fasttext.txt')
    test_fasttext_file = os.path.join(dataset_path, 'test_fasttext.txt')
    convert_to_fasttext_format(train_df, train_fasttext_file)
    convert_to_fasttext_format(test_df, test_fasttext_file)

    # 训练 FastText 模型
    start_time = time.time()
    model = fasttext.train_supervised(train_fasttext_file, epoch=25, lr=0.1, wordNgrams=2)
    train_time = time.time() - start_time
    print(f'{dataset} - Training Time: {train_time:.2f} seconds')

    # 计算测试集上的 Macro-F1 指标
    start_time = time.time()
    true_labels, predictions = evaluate_model(model, test_fasttext_file)
    test_time = time.time() - start_time
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    print(f'{dataset} - Test Macro-F1: {macro_f1:.4f}')
    print(f'{dataset} - Testing Time: {test_time:.2f} seconds')

    # 记录结果
    results.append({
        'Dataset': dataset,
        'Macro-F1': macro_f1,
        'Running Time': train_time+test_time
    })

    # 保存模型
    model.save_model(os.path.join(dataset_path, 'fasttext_model.bin'))

# 保存结果到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('fasttext_results.csv', index=False)

print("fasttext_results.csv generated")

