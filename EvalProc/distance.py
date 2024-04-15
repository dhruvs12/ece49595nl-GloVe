import argparse
import numpy as np

def load_word_vectors():
    parser = argparse.ArgumentParser(description="Load word vectors from files.")
    parser.add_argument('--vocab_path', default='vocab.txt', help='Path to vocabulary file.', type=str)
    parser.add_argument('--vector_path', default='vectors.txt', help='Path to vectors file.', type=str)
    args = parser.parse_args()

    with open(args.vocab_path, 'r') as vocab_file:
        vocabulary = {line.strip().split()[0]: idx for idx, line in enumerate(vocab_file)}

    vector_dimensions = None
    vector_matrix = {}
    with open(args.vector_path, 'r') as vector_file:
        for line in vector_file:
            parts = line.strip().split()
            vector = np.array([float(num) for num in parts[1:]])
            if vector_dimensions is None:
                vector_dimensions = len(vector)
            vector_matrix[parts[0]] = vector

    embedding_matrix = np.zeros((len(vocabulary), vector_dimensions))
    for word, index in vocabulary.items():
        if word in vector_matrix:
            embedding_matrix[index, :] = vector_matrix[word]

    # Normalize the word vectors
    norms = np.linalg.norm(embedding_matrix, axis=1)
    embedding_matrix = embedding_matrix / norms[:, None]

    index_to_word = {index: word for word, index in vocabulary.items()}
    return embedding_matrix, vocabulary, index_to_word

def compute_similarity(embedding_matrix, vocabulary, index_to_word, query, top_n=100):
    terms = query.split()
    if any(term not in vocabulary for term in terms):
        print(f"Some words in the query '{query}' are not in the dictionary.")
        return

    # Compute the composite query vector
    query_indices = [vocabulary[term] for term in terms]
    query_vector = np.sum(embedding_matrix[query_indices, :], axis=0)
    query_norm = np.linalg.norm(query_vector)
    normalized_query_vector = query_vector / query_norm

    # Calculate cosine similarity
    similarities = embedding_matrix.dot(normalized_query_vector)
    for idx in query_indices:
        similarities[idx] = -np.inf  # exclude the query words themselves

    top_indices = np.argsort(-similarities)[:top_n]
    print("\nWord\t\tCosine Similarity\n")
    print("---------------------------------\n")
    for idx in top_indices:
        print(f"{index_to_word[idx]}\t\t{similarities[idx]:.6f}")

def main():
    embedding_matrix, vocabulary, index_to_word = load_word_vectors()
    while True:
        user_input = input("Enter a word or phrase ('EXIT' to quit): ")
        if user_input.lower() == 'exit':
            break
        compute_similarity(embedding_matrix, vocabulary, index_to_word, user_input)

if __name__ == "__main__":
    main()
