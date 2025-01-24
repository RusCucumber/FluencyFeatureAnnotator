# Oral Fluency Feature Annotation System

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

Oral fluency feature annotation system is a open-source program which allows users to detect temporal speech features, including disfluency words (e.g., repetitions, self-repairs, & false starts) and pause locations (i.e., mid- & end-clause pauses) and calculate utterance fluency measures (See [Matsuura et al. (2025)](https://doi.org/10.1016/j.rmal.2024.100177) for more detailed information). Python scripts of the system are available here.

## Installation

1. **Install Docker**
    - Following steps in the link, install and set up Docker in your computer.
2. **Pull Docker image**
    - Run the following command in Terminal.
    ```{bash}
    docker pull ruscucumber/fluency-feature-annotator:0.2.0
    ```
3. **Run Docker container**
    - Run the following command in Terminal.
    ```{bash}
    docker run -d -p 8001:8001 -v ~/Downloads:/app/fluencyfeatureannotator/results -it --name ffa ruscucumber/fluency-feature-annotator:0.2.0
    ```
4. **Open Web application**
    - Open the following URL in your browser: [http://127.0.0.1:8001/](http://127.0.0.1:8001/)

## Usage

1. Prepare pairs of `wav` and `txt` files.
    - You need to ensure that the filenames of wav and txt are the same.
    - Remove punctuations and symbols (e.g., `.`, `,`, `!`, `?`, `"`, `&`, `$`, `-`) in txt files.
2. Click "① Select wav & txt files" button and select target files.
3. Click "② Upload wav & txt files" button and upload selected files.
4. Click "③ Annotate fluency features" button.
5. Check your Downloads directory.
    - Annotation results are saved as a TextGrid format in results folder.
    - Utterance fluency measures are saved as "results/result.csv".

## Citation

Please cite the following thesis in your paper.
> Matsuura, R., Suzuki, S., Takizawa, K., Saeki, M., & Matsuyama, Y. (2025). Gauging the Validity of Machine Learning-Based Temporal Feature Annotation to Measure Fluency in Speech Automatically. Research Methods in Applied Linguistics, 4(1), 1–23. [https://doi.org/10.1016/j.rmal.2024.100177](https://doi.org/10.1016/j.rmal.2024.100177)


## Limitations

- To use the current version of the system, you need to ensure at least 5GB of available storage space.
- The current version of the system has only been tested on the following environment. It is not currently guaranteed that the system will work in other environments.

|Component       |Details             |
|----------------|--------------------|
| OS             | macOS Ventura 13.4 |
| RAM            | Apple M1 16GB      |
| Python version | 3.9.6              |


- Since the current version of the system saves the results of annotation as "result.csv", please be careful to avoid overwriting a file with the same name.
- If you input long audio files (e.g., long than 3 mins), the current version of the system may stop automatic annotation due to a memory issue.
- If you would like to use raw python scripts with pre-trained parameters, please contact at rmatsuur[at]andrew.cmu.edu
- If you find any issues, please contact at rmatsuur[at]andrew.cmu.edu or add issues in this repository.

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg