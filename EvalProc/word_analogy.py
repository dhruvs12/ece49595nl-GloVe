import argparse
import numpy as np
import sys

def load_data(vocab_path='vocab.txt', vectors_path='vectors.txt'):
    """Loads vocabulary and vectors from files."""
    with open(vocab_path, 'r') as vf:
        words = [line.strip().split(' ')[0] for line in vf]

    vectors = {}
    with open(vectors_path, 'r') as vf:
        for line in vf:
            parts = line.strip().split()
            vectors[parts[0]] = np.array([float(val) for val in parts[1:]])

    return words, vectors

def prepare_embeddings(words, vectors):
    """Prepare the word vectors matrix and vocab dictionaries."""
    vocab = {word: index for index, word in enumerate(words)}
    reverse_vocab = {index: word for index, word in enumerate(words)}
    dim = len(vectors[next(iter(vectors))])
    embedding_matrix = np.zeros((len(vocab), dim))

    for word, vec in vectors.items():
        if word in vocab:
            embedding_matrix[vocab[word]] = vec

    # Normalize the vectors
    norms = np.linalg.norm(embedding_matrix, axis=1)
    embedding_matrix = embedding_matrix / norms[:, np.newaxis]

    return embedding_matrix, vocab, reverse_vocab

def find_closest_embeddings(embedding_matrix, vocab, reverse_vocab, query, top_n=10):
    """Find the top_n closest terms in the embedding space for the given query."""
    if query not in vocab:
        print(f'Word: {query} not found in the dictionary.')
        return

    query_idx = vocab[query]
    query_vec = embedding_matrix[query_idx]

    # Compute cosine similarity
    similarity = np.dot(embedding_matrix, query_vec)
    # Ignore the query word itself
    similarity[query_idx] = -1

    # Fetch top N indices sorted by similarity
    top_indices = np.argsort(similarity)[::-1][:top_n]
    results = [(reverse_vocab[i], similarity[i]) for i in top_indices]

    print("\nWord\t\tCosine Similarity\n")
    print("----------------------------------\n")
    for word, score in results:
        print(f"{word}\t\t{score:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--vocab_file', default='vocab.txt', type=str, help='Path to vocabulary file.')
    parser.add_argument('--vectors_file', default='vectors.txt', type=str, help='Path to vectors file.')
    args = parser.parse_args()

    words, vectors = load_data(args.vocab_file, args.vectors_file)
    W, vocab, ivocab = prepare_embeddings(words, vectors)

    while True:
        term = input("\nEnter a word or EXIT to stop: ")
        if term.lower() == 'exit':
            break
        find_closest_embeddings(W, vocab, ivocab, term, top_n=100)

if __name__ == "__main__":
    main()

