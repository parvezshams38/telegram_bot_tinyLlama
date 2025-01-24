
import torch
from transformers import pipelinefrom 
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

TOKEN:Final = 'Your_token_from_FatherBot_in_telegram'

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16, device_map="auto")

def handle_response(question:str) -> str:
    messages = []
    messages.append({"role": "user", "content": question})
    print(question) ## to see in the terminal what user asked
    prompt = pipe.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=100,
                  do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
    response = outputs[0]['generated_text'].split('<|assistant|>')[-1].strip()
    
    if not response.endswith(('.', '!', '?')):
        last_punctuation = max(response.rfind('.'), response.rfind('!'), 
                              response.rfind('?'))
        if last_punctuation != -1:
            response = response[:last_punctuation + 1]
        else:
            response += '.'  # Default to a period 
    
    print(response) ## to see what our LLM answers
    return response

async def start_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello and welcome to my bot!!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text: str = update.message.text
    response: str = handle_response(text)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    print("starting")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",start_command))
    app.add_handler(MessageHandler(filters.TEXT,handle_message))
    app.add_error_handler(error)


    print("polling")
    app.run_polling(poll_interval=10)
