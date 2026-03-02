from src import phonology, model

if __name__ == "__main__":
    WORDS, SYLL = phonology.preprocess_languages()
    MODEL, DATA = model.fit_model(SYLL)
