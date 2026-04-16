import os
import shutil
import gradio as gr
from rag_pipeline import build_vectorstore_from_pdf, search, stream_answer

CURRENT_PDF_PATH = "uploaded/current.pdf"
CURRENT_VECTORSTORE_PATH = "uploaded_vectorstore"


def upload_and_index(pdf_file):
    global CURRENT_PDF_PATH

    if pdf_file is None:
        return "Please upload a PDF first."

    os.makedirs("uploaded", exist_ok=True)
    shutil.copy(pdf_file, CURRENT_PDF_PATH)

    status = build_vectorstore_from_pdf(
        CURRENT_PDF_PATH,
        save_path=CURRENT_VECTORSTORE_PATH
    )
    return f"✅ {status}"


def chat_with_pdf(message, history):
    history = history or []

    if not message or not message.strip():
        yield history, ""
        return

    if not os.path.exists(CURRENT_PDF_PATH):
        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": "Please upload and index a PDF first."}
        )
        yield history, ""
        return

    results = search(
        message,
        k=4,
        vectorstore_path=CURRENT_VECTORSTORE_PATH
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, ""

    for partial_answer in stream_answer(message, results):
        history[-1] = {
            "role": "assistant",
            "content": partial_answer
        }
        yield history, ""


def clear_chat():
    return [], ""


custom_css = """
html, body, .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: #07111f !important;
}

footer {
    display: none !important;
}

.app-shell {
    min-height: 100vh;
    padding: 20px;
}

.sidebar {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 18px;
    min-height: calc(100vh - 40px);
}

.main-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: 18px;
    min-height: calc(100vh - 40px);
}

#chatbot {
    height: 500px !important;
    overflow-y: auto !important;
}

.brand-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 6px;
}

.brand-subtitle {
    opacity: 0.8;
    font-size: 14px;
    margin-bottom: 16px;
}

button {
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}


button.primary {
    background: linear-gradient(135deg, #6C63FF, #4A90E2) !important;
    border: none !important;
}



.suggestion-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row(elem_classes="app-shell", equal_height=True):
        with gr.Column(scale=1, min_width=320, elem_classes="sidebar"):
            gr.HTML("""
                <div class="brand-title">🤖 AI Document Assistant</div>
                <div class="brand-subtitle">Upload PDF & chat with it locally</div>
            """)

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            index_button = gr.Button("Build Index", variant="primary")
            index_status = gr.Textbox(label="Status", lines=3, interactive=False)

            gr.Markdown("### Quick Prompts")
            suggestion_1 = gr.Button("Summarize document")
            suggestion_2 = gr.Button("What is the main idea?")
            suggestion_3 = gr.Button("List key points")
            

            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=4, elem_classes="main-panel"):
            gr.HTML("<div class='brand-title'>AI Chat</div>")

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "👋 Upload a PDF and start asking questions!"}],
                elem_id="chatbot",
                height=500
            )

            msg = gr.Textbox(
                placeholder="Ask anything about the document...",
                lines=2,
                show_label=False
            )

            ask_btn = gr.Button("Send ➜", variant="primary")

    index_button.click(upload_and_index, pdf_input, index_status)
    ask_btn.click(chat_with_pdf, [msg, chatbot], [chatbot, msg])
    msg.submit(chat_with_pdf, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, None, [chatbot, msg])

    
    suggestion_1.click(lambda: "Summarize this document", outputs=msg)
    suggestion_2.click(lambda: "What is the main idea of this document?", outputs=msg)
    suggestion_3.click(lambda: "List the key points in this document", outputs=msg)
    
    

demo.launch()