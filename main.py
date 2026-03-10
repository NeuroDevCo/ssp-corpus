from src import languages, words, syllables, model

if __name__ == "__main__":
    LANGS = languages.get_languages()
    WORDS = words.get_words(LANGS)
    SYLLS = syllables.get_syllables(WORDS)
    DATA = model.prepare_data(SYLLS, languages=LANGS)
    MODEL, DATA_MODEL = model.fit_model(DATA)
