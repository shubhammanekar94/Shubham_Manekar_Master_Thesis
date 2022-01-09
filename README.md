IDS_using_XAI
==============================

This project presents a novel approach of XAI as feature selection technique and the comparative analysis with traditional featrue selection techniques.  

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.│
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── main.py        <- Main python file to regulate the flow of the code
    │   │
    │   ├── data
    │   │  ├── processed      <- The final, canonical data sets for modeling.
    │   │  └── raw            <- The original, immutable data dump.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── Logs           <- Generated Log Files to keep track of the flow of the code
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   └── model.py                
    │   │
    │   ├── preprocessing  <- Script to develope the preprocessing pipeline
    │   │   └── preprocessing.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── visualize.py   
            └── Figures    <- Generated graphics and figures to be used in reporting
    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
