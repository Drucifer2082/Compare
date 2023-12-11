import fitz  
import gensim.corpora
import gensim.models
from gensim.parsing.preprocessing import preprocess_string
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text("text")
        return text


def preprocess_string(text):
    """Preprocess the text for further analysis."""
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    tokens = word_tokenize(text.strip())

    return tokens


def perform_analyses(text1, text2):
    """Perform all analyses on the provided texts."""
    perform_ngram_analysis(text1, text2)
    perform_dependency_parsing(text1, text2)
    perform_word_embedding_analysis(text1, text2)
    perform_ner(text1, text2)
    perform_paragraph_level_sentiment_analysis(text1, text2)
    perform_plagiarism_detection(text1, text2)


def perform_ngram_analysis(text1, text2):
    """Perform n-gram analysis on the provided texts."""
    ngrams = nltk.ngrams(preprocess_string(text1), 2)
    for ngram in ngrams:
        if ngram in preprocess_string(text2):
            print(f"N-gram: {ngram}")


def perform_dependency_parsing(text1, text2):
    """Perform dependency parsing on the provided texts."""
    dependencies1 = nltk.DependencyParser().parse(
        preprocess_string(text1).split())
    dependencies2 = nltk.DependencyParser().parse(
        preprocess_string(text2).split())

    for (word1, word2, tag) in dependencies1:
        for (word2, word3, tag) in dependencies2:
            if word1 == word2:
                print(f"Dependency: {(word1, word2, tag)}")


def perform_word_embedding_analysis(text1, text2):
    """Perform word embedding analysis on the provided texts."""
    doc2vec_model = gensim.models.Doc2Vec.load("doc2vec.model")

    doc1_vec = doc2vec_model.docvecs[text1]
    doc2_vec = doc2vec_model.docvecs[text2]

    similarity = cosine_similarity(doc1_vec, doc2_vec)

    if similarity > 0.7:
        print(f"Word Embedding Similarity: {similarity}")


def perform_ner(text1, text2):
    """Perform NER on the provided texts."""
    entities1 = ne_chunk(
        pos_tag(preprocess_string(text1).split()))
    entities2 = ne_chunk(
        pos_tag(preprocess_string(text2).split()))

    for entity in entities1:
        if entity in entities2:
            print(f"Named Entity Recognition: {entity}")


def perform_paragraph_level_sentiment_analysis(text1, text2):
    """Perform paragraph-level sentiment analysis on the provided texts."""
    paragraphs1 = preprocess_string(text1).split("\n\n")
    paragraphs2 = preprocess_string(text2).split("\n\n")

    for paragraph1, paragraph2 in zip(paragraphs1, paragraphs2):
        blob1 = TextBlob(paragraph1)
        blob2 = TextBlob(paragraph2)

        print(f"Paragraph Level Sentiment Analysis: {paragraph1}")
        print(f"Polarity: {blob1.sentiment.polarity:.2f}")
        print(f"Subjectivity: {blob1.sentiment.subjectivity:.2f}")


def perform_plagiarism_detection(text1, text2):
    """
    Perform plagiarism detection using Jaccard similarity.
    """
    vectorizer = CountVectorizer()

    doc1_vec = vectorizer.fit_transform([text1])
    doc2_vec = vectorizer.transform([text2])

    similarity = cosine_similarity(doc1_vec, doc2_vec)

    for i in range(0, similarity.shape[1]):
        if similarity[i][0] > 0.7:
            print(
                f"Similarity between {text1} and {text2}: {similarity[i][0]}"
            )


if __name__ == "__main__":
    # Provide the paths to your PDF files
    baseline_pdf = "path/to/baseline.pdf"
    comparison_pdf = "path/to/comparison.pdf"

    # Extract text from the PDFs
    text_baseline = extract_text_from_pdf(baseline_pdf)
    text_comparison = extract_text_from_pdf(comparison_pdf)

    # Perform the analyses
    perform_analyses(text_baseline, text_comparison)
