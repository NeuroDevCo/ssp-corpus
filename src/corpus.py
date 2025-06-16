"""Preprocess CHILDES corpora."""

from pathlib import Path
import re
from collections import Counter
import pandas as pd
from rich.progress import track
import pylangacq
import eng_to_ipa as ipa

path = Path("data", "CHILDES", "Bates", "Free20")
files = list(path.glob("*.cha"))

all_tokens = []
errors = []

for file in track(files, description="Extracting words"):
    try:
        data = pylangacq.read_chat(str(file))
        tokens = data.words()
        all_tokens.extend(tokens)
    except ValueError as e:
        print(f"Error reading file {file}: {e}")
        errors.append(file)

# Count token frequency
token_counts = Counter(all_tokens)
print(f"\nTotal unique tokens: {len(token_counts)}")

# Initialize lists for the table
tokens, freqs, phon_trans, syll1, syll2 = [], [], [], [], []
no_transcription = []


# Function to split phonetic transcription into syllables (very simplified)
def split_syllables(phonetic):
    return re.split(r"(?<=[aeiouɑɛɪʊɔəæʌ])(?=[^aeiouɑɛɪʊɔəæʌ])", phonetic)


# Build the table
for token, freq in token_counts.items():
    tokens.append(token)
    freqs.append(freq)

    try:
        phon = ipa.convert(token)
    except:
        phon = ""

    if phon.strip() == "":
        no_transcription.append(token)

    phon_trans.append(phon)
    if phon.strip():
        syls = split_syllables(phon)
        syll1.append(syls[0] if len(syls) > 0 else "")
        syll2.append(syls[1] if len(syls) > 1 else "")
    else:
        syll1.append("")
        syll2.append("")

# Create DataFrame
df = pd.DataFrame(
    {
        "Token": tokens,
        "Frequency": freqs,
        "Phon_Transcription": phon_trans,
        "First_Syllable": syll1,
        "Second_Syllable": syll2,
    }
)

# Sort by frequency
df = df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# Save complete table to Excel
excel_name = "ALL_tokens_phon_FULL.xlsx"
df.to_excel(excel_name, index=False)

# Save tokens without transcription separately
missing_path = "tokens_without_transcription.csv"
pd.DataFrame({"Token": no_transcription}).to_csv(
    missing_path, index=False, encoding="utf-8-sig"
)

# Print summary
print("\nProcessing complete.")
print(df.head(10))
print(f"\nTotal files processed: {len(files)}")
print(f"Files skipped due to errors: {len(errors)}")
print(f"Words without transcription: {len(no_transcription)}")
