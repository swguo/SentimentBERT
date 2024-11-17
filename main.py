# main.py
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_processing import load_data, preprocess_data
from dataset import create_data_loader
from model import Classifier
from train import train_epoch, eval_model

def main(data_path, model_name, max_len, batch_size, epochs, learning_rate):
    # 設定裝置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 資料載入與前處理
    df = load_data(data_path)
    df = preprocess_data(df)

    # 切分資料集（80% 訓練集，20% 測試集）
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 保存訓練集和測試集至 CSV
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)

    # Tokenizer 設定
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 資料加載
    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    # 模型建立
    model = Classifier(n_classes=3, model_name=model_name)
    model = model.to(device)

    # 設定訓練參數
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 訓練和評估
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    best_accuracy = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_test))

        print(f"Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

        # 將張量轉換為 CPU 上的數值
        history["train_acc"].append(train_acc.cpu().numpy())
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc.cpu().numpy())
        history["val_loss"].append(val_loss)
        
        # 儲存最佳模型
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    # 繪製訓練和驗證歷史
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT sentiment model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="Pretrained model name.")
    parser.add_argument('--max_len', type=int, default=160, help="Max token length.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=4, help="Number of epochs.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate.")
    
    args = parser.parse_args()
    main(args.data_path, args.model_name, args.max_len, args.batch_size, args.epochs, args.learning_rate)
