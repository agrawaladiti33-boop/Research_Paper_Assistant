import chromadb
from chromadb.utils import embedding_functions

class VectorManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # We use the same model as before
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            embedding_function=self.embedding_function
        )

    def add_chunks(self, chunks, paper_name):
        """
        Adds text chunks to the ChromaDB collection with paper_name metadata.
        """
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append({
                "page": chunk["page"],
                "paper_name": paper_name
            })
            ids.append(f"{paper_name}_chunk_{i}")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def get_uploaded_papers(self):
        """
        Returns a sorted list of unique paper names in the database.
        """
        results = self.collection.get(include=["metadatas"])
        papers = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                papers.add(meta["paper_name"])
        return sorted(list(papers))

    def _count_paper_chunks(self, paper_name):
        """Returns the number of chunks stored for a specific paper."""
        results = self.collection.get(
            where={"paper_name": paper_name},
            include=["metadatas"]
        )
        return len(results["metadatas"]) if results and results["metadatas"] else 0

    def query(self, query_text, paper_name=None, n_results=5):
        """
        Searches the vector database for the most relevant chunks.
        Safely handles per-paper chunk count to avoid ChromaDB errors.
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []

        where_clause = {"paper_name": paper_name} if paper_name else None

        # Cap n_results to the actual number of available chunks for this filter
        if paper_name:
            available = self._count_paper_chunks(paper_name)
        else:
            available = total_count

        if available == 0:
            return []

        safe_n = min(n_results, available)

        search_results = self.collection.query(
            query_texts=[query_text],
            n_results=safe_n,
            where=where_clause
        )

        retrieved = []
        if search_results["documents"] and search_results["documents"][0]:
            for text, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
                retrieved.append({
                    "text": text,
                    "page": meta["page"],
                    "paper_name": meta["paper_name"]
                })
        return retrieved

    def query_multiple_papers(self, query_text, paper_names, n_results_per_paper=5):
        """
        Retrieves relevant chunks from each specified paper separately.
        Returns: dict {paper_name: [chunks]}
        """
        results = {}
        for name in paper_names:
            results[name] = self.query(query_text, paper_name=name, n_results=n_results_per_paper)
        return results
