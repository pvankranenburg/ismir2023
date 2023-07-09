This repository contains code and supplementary material for: P. van Kranenburg and E. Kearns, “Algorithmic Harmonization of Tonal Melodies using Weighted Pitch Context Vectors”, in Proc. of the 24th Int. Society for Music Information Retrieval Conf., Milan, Italy, 2023.

## Prerequisites:
- lilypond installed and in command line path.
- convert (ImageMagick) installed and in command line path.

## Running the demo
In root of the repository do:
```
$ poetry install
```
This creates a virtual environment with the pitchcontext package installed.

To run:
First activate the environment:
```
$ streamlit run st_dissonance.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
```
Then run the demo:
```
$ streamlit run demo_harmonization.py
```
The identifier of the melody can be inserted in the 
