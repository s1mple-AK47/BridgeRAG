import openai
import numpy as np

# Point to the local vLLM server
client = openai.OpenAI(
    base_url="http://localhost:8999/v1",
    api_key="not-needed"
)

def get_embedding(text: str, model: str = "nomic-embed-text-v1.5"):
    """
    Get embedding for a given text using the local vLLM server.
    """
    try:
        embedding_response = client.embeddings.create(
            model=model,
            input=[text]
        )
        embedding = embedding_response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    sample_text = "This is a test sentence for the embedding model."
    print(f"Getting embedding for: '{sample_text}'")

    embedding_vector = get_embedding(sample_text)

    if embedding_vector:
        # Convert to numpy array for easier handling
        embedding_np = np.array(embedding_vector)
        print("Successfully retrieved embedding.")
        print(f"Embedding dimension: {embedding_np.shape[0]}")
        print(f"First 5 elements of the embedding: {embedding_np[:5]}")
    else:
        print("Failed to retrieve embedding.")
