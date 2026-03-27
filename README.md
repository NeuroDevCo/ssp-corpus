# ssp-corpus

Estimating the prevalence of the Sonority Sequencing Principle (SSP) across languages.

## Running the code

### Setup

```bash
uv sync
```

### Scraping Wiktionary

For scraping all languages in Glottolog:

```bash
uv run -m src.scraping
```

For one particular language, provide its ISO-639-P3 code (e.g., `jpn` for Japanese):

```bash
uv run -m src.scraping jpn
```