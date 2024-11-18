# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer
from model import Classifier

# 設置模型和參數
model_name = 'bert-base-uncased'
max_len = 160
class_names = ['negative', 'neutral', 'positive']

# 載入模型和 tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Classifier(n_classes=3, model_name=model_name)
model.load_state_dict(torch.load("model/best_model_state.bin", map_location=device))
model = model.to(device)
model.eval()

def predict_sentiment(review):
    # 將輸入文本編碼
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 預測
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        sentiment = class_names[prediction]

    return sentiment

# 建立 Gradio 介面
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Dropdown(choices=["The app is excellent!! ",                    
                  "Not Free. Free upto only 5 Habbits.",
                  "This app is very bad"],
                    label="Select a Review"),
    outputs="text",
    title="Sentiment Analysis",
    description="輸入一篇評論，判別其為正向、中立或負向"
)

# 啟動介面
if __name__ == "__main__":
    interface.launch()
