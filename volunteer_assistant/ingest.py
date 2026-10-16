import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "volunteer_programme"

DOCUMENT_FILES = [
    "volunteer_roles.txt",
    "volunteer_registration.txt",
    "participation_rules.txt",
]

load_dotenv(BASE_DIR / ".env")


def main():
    documents = []

    for filename in DOCUMENT_FILES:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required document is missing: {file_path}"
            )

        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError(f"Document is empty: {filename}")

        documents.append(
            Document(
                page_content=content,
                metadata={"source": filename},
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    # Replace the project's previous chunks when ingestion is rerun.
    # This prevents duplicates and removes outdated document content.
    existing_ids = vector_store.get()["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(
        documents=chunks,
        ids=[f"volunteer-chunk-{i}" for i in range(len(chunks))],
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Database location: {DB_DIR}")
    print("Ingestion complete. You can now run python app.py")


if __name__ == "__main__":
    main()