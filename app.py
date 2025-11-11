from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ========= Model Names =========
CHAT_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
SUMM_MODEL_NAME = "t5-small"

print("✅ Loading Chat Model...")
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    CHAT_MODEL_NAME,
    dtype="auto"
)

print("✅ Loading Summarizer...")
summarizer = pipeline("summarization", model=SUMM_MODEL_NAME)

app = Flask(__name__)

# Store conversation history for context
conversation_history = []

# ========= SYSTEM PROMPT =========
SYSTEM_PROMPT = (
    "You are an intelligent, helpful AI chatbot. "
    "Greet politely when greeted. "
    "Your job is to answer questions clearly and naturally. "
    "If asked about education, career, AI, technology, tutorials, or basic facts — answer briefly. "
    "Avoid asking unnecessary personal questions. "
    "Keep responses short (1–3 sentences). "
    "If unsure, say you are not sure."
)


def build_prompt(history, user_msg):
    """Build full prompt including history."""
    lines = [SYSTEM_PROMPT]

    # Use only last 4 rounds for context
    for turn in history[-4:]:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Chatbot: {turn['bot']}")

    lines.append(f"User: {user_msg}")
    lines.append("Chatbot:")
    return "\n".join(lines)


def generate_bot_reply(prompt):
    """Generate reply from LLM."""
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract reply after last "Chatbot:"
    if "Chatbot:" in text:
        text = text.split("Chatbot:")[-1].strip()

    # Remove extra roles
    for tok in ["User:", "Assistant:", "Human:", "AI:"]:
        if tok in text:
            text = text.split(tok)[0].strip()

    # Filter generic apology answers
    if text.lower().startswith("i'm sorry") or text.lower().startswith("sorry"):
        text = "Could you ask that in another way?"

    # If only a question → rewrite
    if text.endswith("?"):
        text = text.replace("?", ".")
    
    return text or "I'm not sure."


def summarize_long_text(txt):
    """Chunk + summarize safely."""
    words = txt.split()
    chunk_size = 350
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    for ch in chunks:
        sm = summarizer(ch, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
        summaries.append(sm)

    if len(summaries) == 1:
        return summaries[0]

    final = summarizer(" ".join(summaries), max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
    return final


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"response": "Please type something."})

    # Greeting handler
    if user_msg.lower() in ["hi", "hello", "hey", "hii"]:
        bot_reply = "Hello! How can I help you today?"
        conversation_history.append({"user": user_msg, "bot": bot_reply})
        return jsonify({"response": bot_reply})

    prompt = build_prompt(conversation_history, user_msg)
    bot_reply = generate_bot_reply(prompt)

    conversation_history.append({"user": user_msg, "bot": bot_reply})
    return jsonify({"response": bot_reply})


@app.route("/summarize", methods=["GET"])
def summarize():
    if not conversation_history:
        return jsonify({"summary": "No conversation yet."})

    transcript = "\n".join(
        [f"User: {t['user']}\nChatbot: {t['bot']}" for t in conversation_history]
    )

    summary = summarize_long_text(transcript)
    return jsonify({"summary": summary})


@app.route("/reset", methods=["POST"])
def reset():
    conversation_history.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("✅ App running at: http://127.0.0.1:5000")
    app.run(debug=True)
