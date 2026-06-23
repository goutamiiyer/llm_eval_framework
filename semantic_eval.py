from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_semantic_similarity(response: str, expected: str, threshold: float = 0.7) -> dict:
    response_embedding = model.encode(response, convert_to_tensor=True)
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    
    similarity = util.cos_sim(response_embedding, expected_embedding).item()
    similarity = round(similarity, 4)
    
    return {
        "passed": similarity >= threshold,
        "score": similarity,
        "response": response,
        "expected": expected,
        "reason": f"Cosine similarity: {similarity} (threshold: {threshold})"
    }