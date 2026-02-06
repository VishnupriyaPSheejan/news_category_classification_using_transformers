# news_category_classification_using_transformers
News Category Classification using Transformers


## Dataset
We used the **HuffPost News Category Dataset** from Kaggle:  
[News Category Dataset (HuffPost)](https://www.kaggle.com/datasets/rmisra/news-category-dataset?utm_source=chatgpt.com)

- Contains ~210,000 news headlines with short descriptions.
- Each record has a news category label (Politics, Business, Sports, Entertainment, Health, Tech, etc.).
- Dataset is in JSON format.

> **Note:** The full dataset is too large to upload to GitHub. Please download it from Kaggle and place it in the notebook directory or your working folder before running training.

---

## Model Used
- **Pretrained Transformer:** `distilbert-base-uncased` (Hugging Face)
- Fine-tuned for multi-class classification.
- Training used **Hugging Face Trainer API** with the following arguments:

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none"
)
## How to Train the Model

Install dependencies:

pip install torch transformers datasets scikit-learn pandas


Download the dataset from Kaggle and place News_Category_Dataset_v2.json in your working directory.

Run the notebook or train.py to fine-tune the model:

trainer.train()


The model and labels will be saved in the model/ folder:

model.save_pretrained("model")
tokenizer.save_pretrained("model")

with open("labels.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

Backend API Instructions

The backend is implemented using FastAPI.
It exposes one endpoint:

POST /predict

Request Body:

{
  "text": "News headline or article text"
}


Response Body:

{
  "predicted_category": "Business",
  "confidence": 0.87
}

## To Run the API:
pip install fastapi uvicorn
python backend.py


Optional: You can use ngrok to expose a public URL for demo/testing:

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

## SQLite Database

Database file: predictions.db

Table name: predictions

Schema:

Column	Type	Description
id	INTEGER	Auto-increment primary key
input_text	TEXT	Text received from API request
predicted_category	TEXT	Model prediction
confidence	REAL	Model confidence (optional)
created_at	TEXT	Timestamp of the request

Initialization is handled automatically when the backend starts:

conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT,
    predicted_category TEXT,
    confidence REAL,
    created_at TEXT
)
""")
conn.commit()
