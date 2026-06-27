library(here)
library(dplyr)
library(childesr)
library(childeswordfreq)
library(arrow)
library(readxl)
library(cli)
library(janitor)
library(purrr)

words <- read_parquet(here("out", "words.parquet"))
langs <- unique(words$iso639p3code)

for (lang in langs) {
  w <- filter(words, iso639p3code == lang)

  if (nrow(w) == 0) {
    next
  }

  out_file <- here(
    "out",
    "childes-freqs",
    paste0("00_childes-freqs_", lang, ".xlsx")
  )

  tryCatch(
    {
      suppressMessages({
        word_counts(
          words = w$words,
          output_file = out_file,
          language = lang,
          role_exclude = c("CHI"),
          per = 1e6,
          zipf = TRUE,
          freq_ignore_special = TRUE,
        )
      })
      cli_alert_success(paste0(lang, ": CHILDES frequencies extracted"))
    },
    error = function(msg) {
      cli_alert_warning(paste0(lang, ": no frequencies found in CHILDES"))
    }
  )
}

files <- list.files(
  here("out", "childes-freqs"),
  pattern = "xlsx",
  full.names = TRUE
)

freqs <- files |>
  set_names(gsub("childes-freqs_|.xlsx", "", basename(files))) |>
  map_dfr(read_xlsx, .name_repair = make_clean_names, .id = "lang") |>
  select(lang, word, freq = total_zipf)

write_parquet(freqs, here("out", "frequencies-childes.parquet"))
