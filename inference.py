# inference.py

import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import GPReviewDataset
from model import Classifier

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),  # 假設 `sentiment` 列在測試集中存在，否則可用 `np.zeros` 填充
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)

def inference(test_data_path, model_path, output_path, model_name='bert-base-uncased', max_len=160, batch_size=16):
    # 設定裝置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 載入模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Classifier(n_classes=3, model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 載入測試資料
    df_test = pd.read_csv(test_data_path)

    # 創建 DataLoader
    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    targets = []
    predictions = []
    review_texts = []
    
    # 推論過程
    with torch.no_grad():
        for d in test_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            targets.extend(d["targets"].cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            review_texts.extend(d["review_text"])

    # 將結果保存至 DataFrame 並輸出至 CSV
    df_results = pd.DataFrame({
        "review_text": review_texts,
        "prediction": predictions,
        "targets" : targets
    })
    
    # 將數字標籤轉換為文字標籤
    #class_names = ['negative', 'neutral', 'positive']
    #df_results['prediction'] = df_results['prediction'].apply(lambda x: class_names[x])

    df_results.to_csv(output_path, index=False)
    print(f"Inference results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on test data.")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to test data CSV.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model.")
    parser.add_argument('--output_path', type=str, default="infr_result.csv", help="Path to save inference results.")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="Pretrained model name.")
    parser.add_argument('--max_len', type=int, default=160, help="Max token length.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size.")

    args = parser.parse_args()
    inference(
        test_data_path=args.test_data_path,
        model_path=args.model_path,
        output_path=args.output_path,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size
    )
