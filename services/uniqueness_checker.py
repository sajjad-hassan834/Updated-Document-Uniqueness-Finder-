async def check_uniqueness(embedding: list[float], collection) -> tuple[float, str]:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    if not results["distances"] or not results["distances"][0]:
        score = 100  
    else:
        score = min(results["distances"][0][0] * 100, 100) 
    score = round(score, 2)
    return score, "good" if score >= 50 else "not good"