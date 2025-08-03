# ssp-corpus

A cross-linguistic corpus analysis on the prevalence of the Sonority Sequencing Principle (SSP) in infant directed speech.

## Repository organization

## Running the code

Making wikipron work in Windows is a bit of a hassle due to some issue with character encoding. If you want to run the dataset acquisition scripts, I recommend you use Linux or macOS or use the Windows Subsystem for Linux (WSL). Otherwise, Windows should be fine.

Run the following commands in your terminal/command prompt (you may need to first install `uv`):

```bash
uv sync
plotly_get_chrome
uv run main.py
```