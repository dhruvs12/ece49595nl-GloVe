import argparse
import numpy as np

def load_and_process_data(args):
    """Loads and processes vocabulary and vectors from specified files."""
    with open(args.vocab_file, 'r') as vocab_file:
        words = [line.strip().split()[0] for line in vocab_file]
    
    vectors = {}
    with open(args.vectors_file, 'r') as vector_file:
        for line in vector_file:
            parts = line.strip().split()
            vectors[parts[0]] = [float(num) for num in parts[1:]]

    return words, vectors

def create_normalized_embeddings(words, vectors):
    """Creates and normalizes the embeddings."""
    vocab = {word: index for index, word in enumerate(words)}
    reverse_vocab = {index: word for index, word in enumerate(words)}

    dim = len(vectors[next(iter(vectors))])
    embeddings = np.zeros((len(vocab), dim))
    for word, vector in vectors.items():
        if word in vocab:
            embeddings[vocab[word]] = vector

    # Normalize embeddings to unit length
    norms = np.linalg.norm(embeddings, axis=1)
    normalized_embeddings = embeddings / norms[:, np.newaxis]

    return normalized_embeddings, vocab, reverse_vocab

def evaluate_embeddings(embeddings, vocab, reverse_vocab):
    """Evaluates the embeddings on predefined tasks."""
    tasks = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt'
    ]
    data_dir = './EvalProc/DataQ/'

    split_size = 100  # Adjust based on memory capacity

    results = {
        'semantic': {'correct': 0, 'total': 0},
        'syntactic': {'correct': 0, 'total': 0},
        'overall': {'correct': 0, 'total': 0, 'seen': 0}
    }

    for task_file in tasks:
        with open(f'{data_dir}{task_file}', 'r') as file:
            full_data = [line.strip().split() for line in file]
            results['overall']['seen'] += len(full_data)
            valid_data = [x for x in full_data if all(word in vocab for word in x)]

        if not valid_data:
            print(f"ERROR: No valid vocab found for {task_file}!")
            continue

        indices = np.array([[vocab[word] for word in row] for row in valid_data])
        pred_vecs = embeddings[indices[:, 1]] - embeddings[indices[:, 0]] + embeddings[indices[:, 2]]
        predictions = np.argmax(np.dot(embeddings, pred_vecs.T), axis=0)

        correct_answers = indices[:, 3] == predictions
        correct_count = np.sum(correct_answers)
        total_count = len(correct_answers)
        results['overall']['total'] += total_count
        results['overall']['correct'] += correct_count

        if tasks.index(task_file) < 5:
            results['semantic']['total'] += total_count
            results['semantic']['correct'] += correct_count
        else:
            results['syntactic']['total'] += total_count
            results['syntactic']['correct'] += correct_count

        print(f"{task_file}: ACCURACY TOP1: {100 * correct_count / total_count:.2f}% ({correct_count}/{total_count})")

    print(f"Questions seen/total: {100 * results['overall']['total'] / results['overall']['seen']:.2f}%")
    print(f"Semantic accuracy: {100 * results['semantic']['correct'] / results['semantic']['total']:.2f}%")
    print(f"Syntactic accuracy: {100 * results['syntactic']['correct'] / results['syntactic']['total']:.2f}%")
    print(f"Total accuracy: {100 * results['overall']['correct'] / results['overall']['total']:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate word vector embeddings.")
    parser.add_argument('--vocab_file', type=str, default='vocab.txt', help='File containing vocabulary.')
    parser.add_argument('--vectors_file', type=str, default='vectors.txt', help='File containing word vectors.')
    args = parser.parse_args()

    words, vectors = load_and_process_data(args)
    embeddings, vocab, reverse_vocab = create_normalized_embeddings(words, vectors)
    evaluate_embeddings(embeddings, vocab, reverse_vocab)

if __name__ == "__main__":
    main()
