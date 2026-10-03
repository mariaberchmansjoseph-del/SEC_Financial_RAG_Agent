import pandas as pd
import json

df = pd.read_csv('data/processed/parsed_documents.csv')
print(f'Parsed documents: {len(df)}')
print(f'Word count mean:  {df["word_count"].mean():.0f}')
print(f'Word count max:   {df["word_count"].max():.0f}')
print()

chunks = []
with open('data/processed/chunks/all_chunks.jsonl',
          encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))
print(f'Total chunks in file: {len(chunks)}')
print()

print('Total words by company:')
by_company = df.groupby('company')['word_count'].sum()
print(by_company.sort_values(ascending=False).to_string())
print()

print('Chunks by company:')
from collections import Counter
chunk_companies = Counter(c['company'] for c in chunks)
for company, count in chunk_companies.most_common():
    print(f'  {company}: {count}')
print()
print('Checking CSV text length vs word count:')
for _, row in df.head(5).iterrows():
    text = str(row.get('full_text', ''))
    words = row['word_count']
    chars = len(text)
    print(f"  {row['company']}: {words} words, "
          f"{chars} chars in CSV")