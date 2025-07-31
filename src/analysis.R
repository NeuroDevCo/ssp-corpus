library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(tidytext)
library(here)
library(ISOcodes)
library(janitor)
library(stringr)

theme_set(
  theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white", colour = "white"),
      plot.background = element_rect(fill = "white", colour = "white")
    )
)

lang_lookup <- ISO_639_3 |>
  clean_names() |>
  as_tibble() |>
  select(lang = id, name, family) |>
  mutate(
    family = str_to_sentence(na_if(family, "")),
    family = if_else(is.na(family), "Unknown", family)
  )

fn <- list.files(here("data", "processed"), full.names = TRUE, pattern = ".csv")
dat <- fn |>
  map(\(x) {
    read_csv(x, show_col_types = FALSE) |>
      drop_na()
  }) |>
  set_names(gsub(".csv", "", basename(fn))) |>
  bind_rows(.id = "lang") |>
  left_join(lang_lookup, by = join_by(lang)) |>
  mutate(
    sonority = strsplit(sonority, ", "),
    family = if_else(is.na(family), "Unknown", family)
  )

fn <- list.files(here("data", "raw"), full.names = TRUE, pattern = ".tsv")
raw <- fn |>
  set_names(gsub(".tsv", "", basename(fn))) |>
  map(\(x) {
    read_csv(x, show_col_types = FALSE, col_names = c("word")) |>
      drop_na(word) |>
      separate(word, c("word", "phon"), sep = "\\t")
  }) |>
  bind_rows(.id = "lang") |>
  left_join(lang_lookup, by = join_by(lang)) |>
  mutate(family = if_else(is.na(family), "Unknown", family))


write_csv(raw, here("out", "pronunciations.csv"))

raw |>
  count(lang, name, family) |>
  arrange(-n) |>
  inner_join(
    dat |> count(lang) |> rename(n_syll = n),
    by = join_by(lang)
  ) |>
  ggplot(aes(n, n_syll, label = lang, colour = family)) +
  geom_text(size = 3) +
  labs(
    x = "Number of tokens",
    y = "Number of distinct syllables",
    colour = "Language family"
  ) +
  scale_x_continuous(
    labels = \(x) format(x, big.mark = ","),
  ) +
  scale_y_continuous(
    labels = \(x) format(x, big.mark = ","),
  ) +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    legend.justification = "right",
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 7),
    axis.title.y = element_blank()
  )

fn <- list.files(here("data", "processed"), full.names = TRUE, pattern = ".csv")
dat <- fn |>
  map(\(x) {
    read_csv(x, show_col_types = FALSE) |>
      drop_na()
  }) |>
  set_names(gsub(".csv", "", basename(fn))) |>
  bind_rows(.id = "lang") |>
  mutate(sonority = strsplit(sonority, ", ")) |>
  left_join(lang_lookup, by = join_by(lang))

write_csv(dat, here("out", "ssp.csv"))

dat |>
  distinct(family, lang) |>
  count(family) |>
  arrange(-n) |>
  ggplot(aes(x = reorder(family, desc(n)), y = n)) +
  geom_col() +
  geom_text(aes(y = n + 3, label = n), size = 3) +
  labs(x = "Language family", y = "Number of languages") +
  theme(
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 35, hjust = 1),
    axis.ticks.x = element_line(size = 3 / 4),
    axis.title.x = element_blank()
  )

ggsave(here("img", "families.png"), height = 5, width = 7, dpi = 800)

dat |>
  count(lang, family, struct) |>
  arrange(lang, n) |>
  group_by(lang) |>
  top_n(10) |>
  ggplot(aes(reorder_within(struct, desc(n), lang), n, fill = family)) +
  facet_wrap(~ reorder(lang, family), scales = "free") +
  geom_col(width = 1, colour = "white") +
  labs(
    x = "Syllabic structure (Top 15)",
    y = "Number of distinct syllables",
    fill = "Language family"
  ) +
  scale_x_reordered() +
  scale_y_continuous(labels = as.integer) +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    legend.justification = "right",
    axis.text.x = element_text(angle = 90, size = 5, hjust = 1),
    axis.text.y = element_text(size = 5),
    panel.grid.major.x = element_blank(),
  )

ggsave(here("img", "syllables.png"), height = 15, width = 15, dpi = 800)

dat |>
  summarise(ssp_prop = mean(is_ssp), .by = c(lang, name, family)) |>
  arrange(ssp_prop) |>
  mutate(
    lang = paste0(
      lang,
      " (",
      name,
      ", ",
      scales::label_percent(0.01)(ssp_prop),
      ")"
    )
  ) |>
  ggplot(aes(ssp_prop, reorder(lang, desc(ssp_prop)), colour = family)) +
  geom_segment(aes(
    x = 0,
    xend = ssp_prop,
    y = reorder(lang, desc(ssp_prop)),
    yend = reorder(lang, desc(ssp_prop))
  )) +
  geom_point() +
  labs(
    x = "Proportion of SSP-following syllabic structures",
    y = "Language",
    colour = "Language family"
  ) +
  scale_x_continuous(labels = scales::percent) +
  theme(
    legend.title = element_blank(),
    legend.position = "top",
    legend.justification = "right",
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 7),
    axis.title.y = element_blank(),
  )

ggsave(here("img", "ssp.png"), height = 15, width = 8, dpi = 800)
