import sys
sys.path.insert(0, '.')
import pandas as pd

df   = pd.read_csv('data/processed/parsed_documents.csv')
nvda = df[(df['ticker'] == 'NVDA') & (df['form_type'] == '10-K')]

print(f'NVIDIA 10-K filings: {len(nvda)}')
for _, row in nvda.iterrows():
    print(f"  {row['filing_date']}: {row['word_count']} words")
    print(f"  Has 1A: {row['has_item_1a']}")
    print(f"  Item 1A preview: {str(row['item_1a'])[:300]}")
    print(f"  Full text preview: {str(row['full_text'])[:300]}")
    print()