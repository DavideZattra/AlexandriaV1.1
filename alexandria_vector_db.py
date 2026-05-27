
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from alexandria_chunking import load_and_chunk_data

# --- CONFIGURAZIONE ARCHITETTURA ---
CHROMA_PATH = "./alexandria_db"
# Usiamo un modello di embedding ultra-rapido, leggero e standard per il RAG locale
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

def build_vector_db():
    print("--- 🏛️ ALEXANDRIA: VECTOR DATABASE BUILDER ---")
    
    # 1. Recupero dei Chunk
    print("\n📦 1. Avvio processo di estrazione e chunking (lettura dai Markdown)...")
    chunks = load_and_chunk_data()
    
    if not chunks:
        print("❌ Errore: Nessun frammento generato. Verifica i file Markdown.")
        return

    # 2. Pulizia del Database (Previene duplicati se rilanci lo script più volte)
    if os.path.exists(CHROMA_PATH):
        print(f"\n🧹 2. Rilevato database precedente in {CHROMA_PATH}. Pulizia in corso...")
        shutil.rmtree(CHROMA_PATH)

    # 3. Inizializzazione Modello di Embedding (100% Locale)
    print(f"\n🧠 3. Caricamento modello di Embedding: {EMBEDDING_MODEL}")
    print("   (La prima volta scaricherà i pesi del modello, circa ~100MB)")
    # LangChain capirà automaticamente se usare la tua RTX 5070 Ti o la CPU
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Ingestione Vettoriale
    print("\n⚙️ 4. Calcolo dei vettori e salvataggio in ChromaDB (Potrebbe richiedere qualche minuto)...")
    
    # Questa singola istruzione crea il database, genera i vettori e li salva su disco
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"\n🎉 SUCCESSO! Knowledge Base indicizzata.")
    print(f"I dati vettoriali sono stati salvati permanentemente nella cartella: {CHROMA_PATH}")

if __name__ == "__main__":
    build_vector_db()