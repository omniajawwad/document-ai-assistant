import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


def load_documents():
    docs = []
    folder_path = "data"

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            print(f"Loading file: {file_path}")
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            print(f"Loaded pages from {file}: {len(loaded_docs)}")
            docs.extend(loaded_docs)

    print(f"Total loaded pages: {len(docs)}")

    if not docs:
        raise ValueError("No PDF documents were loaded from the data folder.")

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("First chunk preview:")
        print(chunks[0].page_content[:300])

    return chunks


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(chunks, save_path="vectorstore"):
    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore


def search(query, k=4, vectorstore_path="vectorstore"):
    embeddings = create_embeddings()
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=k)
    return results


def build_context(results):
    context = []

    for i, r in enumerate(results, 1):
        source = os.path.basename(r.metadata.get("source", "unknown"))
        page = r.metadata.get("page", "unknown")

        if isinstance(page, int):
            page_display = page + 1
        else:
            page_display = page

        context.append(
            f"Source {i} | File: {source} | Page: {page_display}\n{r.page_content}"
        )

    return "\n\n".join(context)


def build_vectorstore_from_pdf(pdf_path, save_path="vectorstore"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)

    return f"Indexed {len(docs)} pages into {len(chunks)} chunks."


def generate_answer(query, results):
    context = build_context(results)

    prompt = f"""
You are a helpful company knowledge base assistant.

Answer the user's question using ONLY the context below.
If the answer is not in the context, say: I could not find that in the provided documents.

Be clear, concise, and professional.
At the end, add:
Sources used: mention the page numbers used only, like: Page 2, Page 4.

Context:
{context}

Question:
{query}

Answer:
"""

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0
    )

    response = llm.invoke(prompt)
    return response.content


def stream_answer(query, results):
    context = build_context(results)

    prompt = f"""
You are a helpful company knowledge base assistant.

Answer the user's question using ONLY the context below.
If the answer is not in the context, say: I could not find that in the provided documents.

Keep the answer concise and structured using bullet points.
Use short, direct bullets whenever possible.
At the end, add:
Sources used: mention the page numbers used only, like: Page 2, Page 4.

Context:
{context}

Question:
{query}

Answer:
"""
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0
    )

    full_text = ""
    for chunk in llm.stream(prompt):
        if hasattr(chunk, "content") and chunk.content:
            full_text += chunk.content
            yield full_text


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)

    query = "How much paid sick leave do employees get and how can they use it?"
    print("\nSearching...\n")
    results = search(query, k=4)

    print("\nGenerating final answer...\n")
    answer = generate_answer(query, results)

    print("Final Answer:\n")
    print(answer)