import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim.models.word2vec import PathLineSentences
from gensim.utils import simple_preprocess
import os
import urllib.request
import zipfile
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from nltk import pos_tag
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger_eng')

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A dog barks at the playful cat in the garden",
    "The cat chases mice while the fox watches",
    "Animals play together in nature's playground",
    "Dogs and cats can become friendly companions"
]

# Preprocess the corpus
processed_corpus = [simple_preprocess(doc) for doc in corpus]

# Create Word2Vec model
model = Word2Vec(
    sentences=processed_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=0,
    epochs=10
)

# Find similar words (using a word from our corpus)
print("\nWords similar to 'fox':")
print(model.wv.most_similar('fox', topn=5))

# Calculate similarity between two thematically related words
print("\nSimilarity between 'fox' and 'dog':")
print(model.wv.similarity('fox', 'dog'))

# Word vector arithmetic with corpus-related words
print("\nVector arithmetic: playful - lazy + quick =")
try:
    print(model.wv.most_similar(positive=['playful', 'quick'], negative=['lazy'], topn=1))
except KeyError:
    print("Error!!")

# Part Two: Download and process Text8 dataset
dataset_url = "http://mattmahoney.net/dc/text8.zip"
dataset_path = "text8.zip"
corpus_file = "text8"

if not os.path.exists(corpus_file):
    print("Downloading the Text8 dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Unzipping the dataset...")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall()
    print("Dataset downloaded and extracted.")

# Train model on Text8 dataset
sentences = PathLineSentences(corpus_file)
large_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=5
)

# King - Man + Woman experiment
print("\nVector arithmetic: king - man + woman =")
try:
    print(large_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))
except KeyError:
    print("Word not in vocabulary!")

words = ['dog', 'cat', 'fox', 'animal', 'lazy', 'playful', 'quick', 'friendly', 'nature', 'garden']
word_vectors = [large_model.wv[word] for word in words if word in large_model.wv]
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    if word in large_model.wv:
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02, word)
plt.title("Word2Vec Word Embeddings Visualization")
plt.show()

# Part Three: Generate meaningful sentence
def generate_simple_sentence(model, seed_word, temperature=0.7):
    """
    Generates grammatically structured sentences using Word2Vec similarities.
    
    Key components:
    1. Templates: Pre-defined sentence structures for natural language generation
    2. POS tagging: Ensures words are used in grammatically correct positions
    3. Temperature sampling: Controls randomness in word selection
    4. Word tracking: Prevents word repetition
    5. Context maintenance: Uses word similarities for coherent sentences
    """
    
    # Helper function to select similar words based on context and POS
    def get_similar_word(word, exclude_words, pos_type=None):
        """
        Selects similar words while considering:
        - Word similarity scores
        - Part of speech requirements
        - Previously used words (to avoid repetition)
        - Temperature-based sampling for controlled randomness
        """
        similar_words = model.wv.most_similar(word, topn=30)
        filtered_words = [(w, s) for w, s in similar_words 
                        if w not in exclude_words and len(w) > 2]
        
        if pos_type:
            # Try to get words with correct POS tag
            filtered_pos_words = [(w, s) for w, s in filtered_words 
                                if pos_tag([w])[0][1] == pos_type]
            if filtered_pos_words:
                filtered_words = filtered_pos_words
        
        if not filtered_words:
            return random.choice([w for w, _ in similar_words])
        
        words, scores = zip(*filtered_words)
        scores = np.array(scores)
        scores = np.exp(np.log(scores) / 1.2)
        scores = scores / scores.sum()
        
        return np.random.choice(words, p=scores)

    
    used_words = {seed_word}
    subject = seed_word
    verb = get_similar_word(subject, used_words, 'VERB')
    used_words.add(verb)
    
    adjective = get_similar_word(subject, used_words, 'ADJ')
    used_words.add(adjective)
    
    obj = get_similar_word(verb, used_words)
    used_words.add(obj)
    
    adverb = get_similar_word(verb, used_words, 'ADV')
    
    # Enhanced templates with more natural language patterns
    templates = [
        "the {adjective} {subject} {verb} {adverb} in the {object}",
        "a {adjective} {subject} {verb} {adverb} near the {object}",
        "the {subject} {verb} {adverb} through the {adjective} {object}",
        "many {adjective} {subject} {verb} {adverb} around the {object}"
    ]
    
    # Select and fill template
    template = random.choice(templates)
    return template.format(
        subject=subject,
        verb=verb,
        object=obj,
        adjective=adjective,
        adverb=adverb
    )

print("\nWords similar to 'fox':")
print(model.wv.most_similar('fox', topn=5))

print("\nSimilarity between 'fox' and 'dog':")
print(model.wv.similarity('fox', 'dog'))

print("\nVector arithmetic: playful - lazy + quick =")
try:
    print(model.wv.most_similar(positive=['playful', 'quick'], negative=['lazy'], topn=1))
except KeyError:
    print("Error!!")

print("\nVector arithmetic: king - man + woman =")
try:
    print(large_model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))
except KeyError:
    print("Word not in vocabulary!")

print("\nGenerated sentences:")
for _ in range(3):
    print(generate_simple_sentence(large_model, "food"))
