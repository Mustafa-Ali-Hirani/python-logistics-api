# advanced_rag_agent.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================================
# STEP 1: LOAD RULES (We split paragraphs as documents)
# =====================================================================
def load_port_rules(filepath="port_rules.txt") -> list:
    """Loads and splits port rules into distinct documents/chunks."""
    if not os.path.exists(filepath):
        print(f"[Error] File {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks

# =====================================================================
# STEP 2: HYDE (Hypothetical Document Embeddings) GENERATION
# =====================================================================
def generate_hyde_document(query: str) -> str:
    """Generates a hypothetical compliance document clause answering the query."""
    print(f"\n[HyDE] Generating hypothetical regulatory text for: '{query}'...")
    
    prompt = f"""
    You are an expert customs compliance officer.
    Write a hypothetical excerpt from a port regulations manual that perfectly answers this user query:
    "{query}"
    
    Do not preface your response or mention that this is hypothetical. Write it exactly like a formal, cold regulatory clause.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    hyde_text = completion.choices[0].message.content.strip()
    return hyde_text

# =====================================================================
# STEP 3: naive SEARCH (Simulating retrieval using substring match/simple overlaps)
# =====================================================================
def naive_retrieval(search_anchor: str, corpus: list, top_n=3) -> list:
    """Simulates a fast initial vector-style search using simple keyword overlap."""
    print(f"[Search] Matching candidates using retrieval anchor...")
    scored_corpus = []
    
    search_terms = set(search_anchor.lower().split())
    for chunk in corpus:
        chunk_terms = set(chunk.lower().split())
        overlap = len(search_terms.intersection(chunk_terms))
        scored_corpus.append((overlap, chunk))
        
    scored_corpus.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_corpus[:top_n]]

# =====================================================================
# STEP 4: RE-RANKING WITH LLM-as-a-Judge
# =====================================================================
def rerank_documents(query: str, retrieved_chunks: list) -> list:
    """Re-ranks retrieved documents and filters out chunks with scores under 50%."""
    print("\n[Re-ranker] Re-ranking candidate documents using LLM-as-a-Judge...")
    reranked_chunks = []
    
    for chunk in retrieved_chunks:
        prompt = f"""
        You are a meticulous customs auditor. 
        Evaluate the relevance of this retrieved port regulation chunk to the user's specific query.
        
        User Query: "{query}"
        Retrieved Chunk:
        ---
        {chunk}
        ---
        
        Provide a relevance score from 0 to 100, where 0 is completely irrelevant and 100 is extremely relevant and directly answers the query.
        Return ONLY a single integer number as your output. Do not write text, explanations, or formatting.
        """
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            score = int(completion.choices[0].message.content.strip())
        except Exception:
            score = 0
            
        print(f" -> Scored Chunk (first 30 chars): '{chunk[:30]}...' | Relevance Score: {score}")
        if score >= 50:  # Threshold of 50
            reranked_chunks.append((score, chunk))
            
    reranked_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in reranked_chunks]

# =====================================================================
# STEP 5: AGENTIC QUERY RE-WRITING (The Self-Correction Step)
# =====================================================================
def reformulate_query(original_query: str) -> str:
    """Reformulates a failed query into alternative industry terminology."""
    print("\n[Agentic Loop] Search yielded no relevant results. Attempting query reformulation...")
    
    prompt = f"""
    You are an expert logistics coordinator. 
    A search for the query: "{original_query}" yielded no matches in our port rules database.
    
    Rewrite this query into a highly simplified search term using common alternative industry vocabulary.
    - If the query mentions "power accumulators", translate it to "batteries".
    - If the query mentions "voltage regulators", translate it to "circuit breakers" or "electrical components".
    - Keep the output extremely brief. Only output the rewritten search query.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    new_query = completion.choices[0].message.content.strip()
    print(f"[Agentic Loop] Reformulated query to: '{new_query}'")
    return new_query

# =====================================================================
# STEP 6: FINAL GENERATION
# =====================================================================
def generate_compliance_response(query: str, context_chunks: list):
    """Generates the final compliance response using the context chunks."""
    print("\n[Generation] Compiling final compliance answer...")
    context = "\n\n".join(context_chunks)
    
    prompt = f"""
    You are an expert customs compliance agent.
    Analyze the provided port regulations and answer the user query.
    
    Port Regulations:
    ---
    {context}
    ---
    
    User Query: {query}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    print("\n=== FINAL COMPLIANCE ANSWER ===")
    print(completion.choices[0].message.content)

# =====================================================================
# MAIN RECURSIVE/AGENTIC PIPELINE
# =====================================================================
def run_advanced_rag_pipeline(query: str, max_attempts=2):
    corpus = load_port_rules()
    current_query = query
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n=============================================")
        print(f"ATTEMPT {attempt} (Query: '{current_query}')")
        print(f"=============================================")
        
        # 1. HyDE Anchor
        hyde_anchor = generate_hyde_document(current_query)
        
        # 2. Match Candidates
        candidates = naive_retrieval(hyde_anchor, corpus, top_n=4)
        
        # 3. Score and Filter
        final_context = rerank_documents(current_query, candidates)
        
        if final_context:
            print(f"\n[Success] Found {len(final_context)} relevant chunks on Attempt {attempt}!")
            generate_compliance_response(query, final_context)
            return
        
        # If no chunks passed the threshold and we have attempts left, reformulate
        if attempt < max_attempts:
            current_query = reformulate_query(current_query)
            
    print("\n[Failure] Agentic RAG was unable to locate any compliant regulations after reformulating.")

if __name__ == "__main__":
    # This query contains 'power accumulators', which does not exist in port_rules.txt
    test_query = "Can we store high-capacity power accumulators in transit sheds at Karachi Port?"
    run_advanced_rag_pipeline(test_query)