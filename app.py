import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Target the available hardware (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Universal Hugging Face Identifier - Clean and safe for GitHub and Spaces
model_id = "sshleifer/distilbart-cnn-12-6"

print("Loading model files...")
# 2. Load tokenizer and model weights using the universal ID
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 3. Define the summarization logic
def summarize_text(text):
    if not text.strip():
        return "Please enter some text to summarize."
        
    # Convert text to tokens and move to the selected device
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    
    # Generate summary tokens
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_new_tokens=150, 
            min_length=30, 
            do_sample=False
        )
        
    # Decode back to clean text
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Starting Gradio Web Server...")
# 4. Create and launch your web application interface
demo = gr.Interface(
    fn=summarize_text, 
    inputs=gr.Textbox(lines=10, placeholder="Paste your long text here..."), 
    outputs="text",
    title="Text Summarization with DistilBART",
    description="Enter a long piece of text and get a concise summary using the DistilBART model fine-tuned on CNN/DailyMail dataset."
)

# The standard configuration for smooth server deployment
if __name__ == "__main__":
    demo.launch()
