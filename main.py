from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

TOKEN:Final = '7224677229:AAEOCF8oYTWLnrGdQVhdUrb37Sr-fWCONTg'
BOT_USERNAME:Final = "@Salparbot"

import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16, device_map="auto")

async def start_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello and welcome to my bot!!")

async def help_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Help!")

async def custom_command(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("What is custom!!")

def handle_response(question:str) -> str:
    messages = []
    messages.append({"role": "user", "content": question})
    print(question)
    prompt = pipe.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=100,
                  do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
    response = outputs[0]['generated_text'].split('<|assistant|>')[-1].strip()
    print(response)
    return response

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #message_type: str = update.message.chat.type
    text: str = update.message.text
    #print(text)
    response: str = handle_response(text)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    print("starting")
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",start_command))
    app.add_handler(CommandHandler("help",help_command))
    
    app.add_handler(MessageHandler(filters.TEXT,handle_message))
    app.add_error_handler(error)


    print("polling")
    app.run_polling(poll_interval=10)