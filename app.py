import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory

# 1. Setup
DetectorFactory.seed = 0
model_checkpoint = "facebook/nllb-200-distilled-600M"

# 2. Load Model & Tokenizer
# We use the pipeline for easy inference
translator = pipeline("translation", model=model_checkpoint, tokenizer=model_checkpoint, max_length=400)
tokenizer = translator.tokenizer

# 3. Define Language Codes
LANG_CODES = {
    "English": "eng_Latn",
    "Urdu": "urd_Arab",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Chinese": "zho_Hans",
    "Arabic": "ara_Arab"
}

AUTO_MAP = {
    "en": "eng_Latn",
    "ur": "urd_Arab",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "ara_Arab"
}

# 4. Main Function
def translate_text(text, source_selection, target_selection):
    if not text:
        return ""

    # Clean text
    clean_text = text.replace("\n", " ").strip()
    target_code = LANG_CODES[target_selection]

    # Handle Source
    if source_selection == "Auto Detect":
        try:
            detected_iso = detect(clean_text)
            source_code = AUTO_MAP.get(detected_iso, "eng_Latn")
            label_prefix = f"**[Detected: {detected_iso.upper()} -> {source_code}]**\n\n"
        except Exception:
            source_code = "eng_Latn"
            label_prefix = "**[Detection Failed, assuming English]**\n\n"
    else:
        source_code = LANG_CODES[source_selection]
        label_prefix = ""

    # Force Target Token
    translator.model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
    
    # Translate
    output = translator(text, src_lang=source_code, tgt_lang=target_code)
    return label_prefix + output[0]['translation_text']

# 5. UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Universal Translator - Multilingual Seq2Seq Transformer")
    gr.Markdown("Translate between **English, Urdu, Hindi, French, German, Spanish, Chinese, and Arabic**.")
    
    with gr.Row():
        source_lang = gr.Dropdown(choices=["Auto Detect"] + list(LANG_CODES.keys()), value="Auto Detect", label="Source Language")
        target_lang = gr.Dropdown(choices=list(LANG_CODES.keys()), value="English", label="Target Language")
    
    input_text = gr.Textbox(placeholder="Type text here...", label="Input Text", lines=3)
    output_text = gr.Textbox(label="Translated Text", lines=3, interactive=False)
    
    translate_btn = gr.Button("Translate", variant="primary")
    
    translate_btn.click(fn=translate_text, inputs=[input_text, source_lang, target_lang], outputs=output_text)

# 6. Launch
demo.launch()
