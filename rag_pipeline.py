import os

from dotenv import load_dotenv
from openai import OpenAI
import psycopg2

load_dotenv()

client = OpenAI()
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "mydb"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

class RAGPipeline:
    def __init__(self):
        self._ensure_table()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the database connection."""
        conn.close()

    def _ensure_table(self):
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    source TEXT,
                    embedding vector(1536)
                )
            """)
        conn.commit()

    def ingest_document(self, content: str, source: str):
        """Chunk and store a document."""
        chunks = self._chunk_text(content)
        for chunk in chunks:
            embedding = self._get_embedding(chunk)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks (content, source, embedding) VALUES (%s, %s, %s)",
                    (chunk, source, embedding)
                )
        conn.commit()
        print(f"Ingested {len(chunks)} chunks from {source}")

    def query(self, question: str, top_k: int = 3) -> str:
        """RAG query: retrieve context, then generate answer."""
        # 1. Retrieve relevant chunks
        context_chunks = self._retrieve(question, top_k)

        if not context_chunks:
            return "I don't have information about that."

        # 2. Build prompt with context
        context = "\n\n".join([c['content'] for c in context_chunks])
        sources = list(set([c['source'] for c in context_chunks]))

        # 3. Generate answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Answer questions based on the provided context.
                    If the context doesn't contain the answer, say "I don't have information about that."
                    Be concise and cite which document the info came from."""
                },
                {
                    "role": "user",
                    "content": f"""Context:
                            {context}

                        Question: {question}"""
                }
            ]
        )

        answer = response.choices[0].message.content
        return f"{answer}\n\nSources: {', '.join(sources)}"

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        """Find most relevant chunks."""
        query_embedding = self._get_embedding(query)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE 1 - (embedding <=> %s::vector) > 0.3
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, query_embedding, top_k))
            return [{"content": r[0], "source": r[1], "similarity": r[2]}
                    for r in cur.fetchall()]

    def _chunk_text(self, text: str, size: int = 300, overlap: int = 30) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk = ' '.join(words[start:start + size])
            chunks.append(chunk)
            start += size - overlap
        return chunks

    def _get_embedding(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
