# A primer on artificial intelligence in plant digital phenomics: embarking on the data to insights journey [![Build Status](https://app.travis-ci.com/HarfoucheLab/A-Primer-on-AI-in-Plant-Digital-Phenomics.svg?branch=main)](https://app.travis-ci.com/HarfoucheLab/A-Primer-on-AI-in-Plant-Digital-Phenomics)
----
![split](https://faridnakhle.com/unitus/DigitalPhenomics/githubimages/logo.png?t=1)

This repository is a supplement to the paper, **A primer on artificial intelligence in plant digital phenomics: embarking on the data to insights journey** (submitted to *Trends in Plant Science, 2022*) by Antoine L. Harfouche, Farid Nakhle, Antoine H. Harfouche, Orlando G. Sardella, Eli Dart, and Daniel Jacobson. It aims to train, for the first time, an interpretable by design model to identify and classify cassava plant diseases, and explain its predictions.

Read the accompanying paper [here](https://doi.org) (a link will be available once the paper is published).

The repository contains the python files for our implementation of 'this looks like that' interpretable by design explainable artificial intelligence (X-AI) algorithm. In addition, this repository contains a tutorial hosted in an interactive computational notebook that can run on Google Colab or any other platform supporting Jupyter notebooks, aiming to help you use our implementation of the algorithm to train a cassava plant diseases classifier that is able to explain its predictions.
By the end of the tutorial, you should be able to apply the code and instructions to other datasets relevant to your research or project.

Access the tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarfoucheLab/A-Primer-on-AI-in-Plant-Digital-Phenomics/blob/main/Tutorial.ipynb)

Our code includes unit tests. Travis continuous integration (CI) service was embedded in our repository to run unit tests automatically when code updates are introduced. This provides a safeguard against updates that can break existing functionality by generating a report showing which tests have passed or failed.

We are working to increase the test coverage to cover all of the implemented functions.
To manually run the unit tests, use the following command:
python -m pytest

NB: pytest is required to run the tests. You might install it using pip (pip install pytest).

Finally, we have prepared a set of self-test quizzes in form of multiple-choice questions (MCQs), practices, and exercises to provide you with opportunities to augment your learning by testing the knowledge you have acquired and applying the concepts explained.

Find the corresponding links below:

**Access the MCQs [here](https://forms.gle/jVZHLpViL2ruYyxCA "here")**

**Access the first exercise (for novices)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarfoucheLab/A-Primer-on-AI-in-Plant-Digital-Phenomics/blob/main/Exercise_Novice.ipynb)

**Access the second exercise (for experienced users)** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarfoucheLab/A-Primer-on-AI-in-Plant-Digital-Phenomics/blob/main/Exercise_Advanced.ipynb)

## Citation
----
If you use any part of this code in your research, kindly cite our paper using the bibtex below (bibtex will be updated once the paper is published):

```
@article{
  title={A primer on artificial intelligence in plant digital phenomics: embarking on the data to insights journey},
  author={Antoine L. Harfouche, Farid Nakhle, Antoine H. Harfouche, Orlando G. Sardella, Eli Dart, Daniel Jacobson},
  journal={},
  pages={},
  year={2022},
  publisher={},
  doi={},
}
```