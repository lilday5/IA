import os
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists("chroma_db") and os.listdir("chroma_db"):
    print("Cargando ChromaDB existente...")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    print("Leyendo documentos...")
    documentos = []
    for archivo in os.listdir("documentos_seguridad"):
        ruta = f"documentos_seguridad/{archivo}"
        if archivo.endswith(".pdf"):
            loader = PyPDFLoader(ruta)
            documentos.extend(loader.load())
            print(f"  PDF: {archivo}")
        elif archivo.endswith(".csv"):
            try:
                loader = CSVLoader(file_path=ruta, encoding="utf-8")
                documentos.extend(loader.load())
            except:
                loader = CSVLoader(file_path=ruta, encoding="latin-1")
                documentos.extend(loader.load())
            print(f"  CSV: {archivo}")
    print(f"  → {len(documentos)} documentos cargados")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    print(f"  → {len(chunks)} chunks generados")
    print("Indexando en ChromaDB...")
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
llm = OllamaLLM(model="tutor_analitico")

PROMPT_TEMPLATE = """<|im_start|>system
Eres un Tutor Analítico especializado en seguridad en México. Responde ÚNICAMENTE usando el contexto.
<|im_end|>
<|im_start|>user
Contexto:
{contexto}

Pregunta: {pregunta}
<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

def formatear_docs(docs):
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source','?')}]\n{d.page_content}"
        for d in docs
    )

cadena = (
    {"contexto": retriever | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n¡RAG LISTO! Escribe 'salir' para terminar.\n")
while True:
    pregunta = input("Tu pregunta: ").strip()
    if pregunta.lower() in ("salir", "exit", "quit"):
        break
    if not pregunta:
        continue
    print("\nTutor:")
    for pedacito in cadena.stream(pregunta):
        print(pedacito, end="", flush=True)
    print("\n" + "-"*60)
