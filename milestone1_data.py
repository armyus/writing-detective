import os
import re
import urllib.request
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Gutenberg IDs and Authors
BOOKS = {
    'Arthur Conan Doyle': [
        'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',
        'https://www.gutenberg.org/cache/epub/2852/pg2852.txt'
    ],
    'Jane Austen': [
        'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
        'https://www.gutenberg.org/cache/epub/158/pg158.txt'
    ],
    'Edgar Allan Poe': [
        'https://www.gutenberg.org/cache/epub/2147/pg2147.txt',
        'https://www.gutenberg.org/cache/epub/2148/pg2148.txt'
    ]
}

def clean_gutenberg_text(text):
    """Strips Project Gutenberg boilerplate and basic cleaning."""
    start_match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*', text)
    end_match = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*', text)
    
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, words_per_chunk=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i:i + words_per_chunk]
        if len(chunk) == words_per_chunk:  # Only keep full chunks
            chunks.append(' '.join(chunk))
    return chunks

def main():
    os.makedirs('data', exist_ok=True)
    dataset = []

    for author, urls in BOOKS.items():
        for url in urls:
            logging.info(f"Fetching data for {author} from {url}")
            try:
                response = urllib.request.urlopen(url)
                raw_text = response.read().decode('utf-8', errors='ignore')
                
                clean_text = clean_gutenberg_text(raw_text)
                chunks = chunk_text(clean_text, words_per_chunk=500)
                
                for chunk in chunks:
                    dataset.append({'author': author, 'text': chunk})
                    
                logging.info(f"Generated {len(chunks)} chunks for {author}")
            except Exception as e:
                logging.error(f"Failed to fetch {url}: {e}")

    df = pd.DataFrame(dataset)
    df.to_csv('data/dataset.csv', index=False)
    logging.info(f"Dataset saved to data/dataset.csv with {len(df)} total samples.")

if __name__ == "__main__":
    main()
