This repository contains code, demo, and supplementary material for: P. van Kranenburg and E. Kearns, “Algorithmic Harmonization of Tonal Melodies using Weighted Pitch Context Vectors”, in Proc. of the 24th Int. Society for Music Information Retrieval Conf., Milan, Italy, 2023.

## Prerequisites:
- lilypond installed and in command line path.
- convert (ImageMagick) installed and in command line path.
- Python 3
- Poetry (https://python-poetry.org)

## Running the demo
In root of the repository do:
```
$ poetry install
```
This creates a virtual environment with the pitchcontext package installed.

Activate the environment:
```
$ poetry shell
```

Then run the demo:
```
$ streamlit run demo_harmonization.py
```
The identifier of the melody can be inserted in the "Song ID" input field.
