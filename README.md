# 🧠 Text-to-SQL — Fine-Tuned LLaMA 3.2 1B Model

A fine-tuned version of **Meta’s LLaMA 3.2 1B** model that generates SQL queries from natural language questions.  
This project leverages supervised fine-tuning and **QLoRA** to adapt the model for the Text-to-SQL task.

📦 The model is available on [🤗 Hugging Face](https://huggingface.co/GuruHemendra/llama3.2-1b-text-to-sql/tree/main)

---

## 📌 Overview

This model converts **natural language prompts** into **structured SQL queries**, making it easier to query databases without needing SQL expertise.

The model is fine-tuned using data from:
> [philschmid/gretel-synthetic-text-to-sql](https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql)

### 🔍 Model Details:
- **Base Model**: LLaMA 3.2 1B
- **Fine-Tuning Method**: Supervised fine-tuning using **QLoRA**
- **Dataset**: Gretel synthetic Text-to-SQL
- **Frameworks**: Hugging Face Transformers + PEFT + QLoRA

---

## 🚀 How to Use

You can load and use the model directly with the Hugging Face 🤗 `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "GuruHemendra/llama3.2-1b-text-to-sql"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "List all customers who placed an order in the last 7 days."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

