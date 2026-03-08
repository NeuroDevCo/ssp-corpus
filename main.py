from src import scraping, phonology, model

if __name__ == "__main__":
    LANGS = scraping.get_languages()
    WORDS, SYLL = phonology.preprocess_languages()
    DATA = model.prepare_data(SYLL, languages=LANGS)
    MODEL, DATA_MODEL = model.fit_model(DATA)
