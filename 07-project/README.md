# Project Handoff

You did it! You got selected to review my project, how exciting.

I've been working really hard on this. There is a lot of things I can do better, but I have a MUCH better feel and appreciation for these technologies now.

No more, *WHINING.*

********************************
**Objective**:

- Predict what quality label a wine will be given.

**Why**?

- What matters to make a good wine? Knowing more about these we can direct our business partners what to focus on when they're creating and crafting their wine.

Project Intro

- Found a dataset online with 1.2k records for wine quality.
- Consisting of 12 columns describing a wine; acidity, alcohol %, most importantly all numeric values

********************************
Technical methodology:

- Make data
    - Takes the raw CSV, splits it into train/valid/test split
- Train Data
    - Connects to MLFlow, using hyperopt we train a multi;softmax XGBoost
    - Records all experiments, also records column names of the original dataset back to MLFlow
    - Records val-mlogloss as the metric of choice (multi-class classification)
- Promote Model
    - Searches all experiments, sorts by val-mlogloss and moves that model into Production
- App.py
    - Flask server that takes in a CURL command (path to a file)
********************************

********************************************How will you use this?********************************************

1. They need to install Git on their machine if they don't have it already.
2. They should clone your repository:

- `git clone https://github.com/samlafell/mlops_zoomcamp_sam.git`

3. Then CD into the repo:

- `cd mlops_zoomcamp_sam/07-project`
********************************

********************************A few things to note:********************************

- You should have `aws cli` and `aws configure`  step already completed on the machine you're running on. Test it out by doing `aws s3 ls`, assuming you have ListBuckets access rights in your own bucket
- You need to have Docker and Docker Compose installed on your machine. If your don't, please install them.

********************************
When you're ready to start:

- Download:  <https://sal-wine-quality.s3.us-east-2.amazonaws.com/test.csv>
    - The bucket is publicy available so you should be able to access and download this test file. After running ./start.sh, you will be prompted to direct the program to the location of this file on your machine. This is scoring data.
- Terminal: ``./start.sh``
- Grafana Monitoring:
    - localhost:3000
    - admin/admin I think is how to log in
    - You can see here daily misclassification rate, I'm pretty sure the dashboards should still be there
        - There were no dates, I added 2 wines per date
        - I couldn't do val-mlogloss for this because train/validation did not have dates on them and also in Test there were only 2 wines per day to get this “time series” feel. And val-mlogloss wouldn't work with 2 wines.
- Postgres:
    - localhost:5432 (But in Grafana I used db:5432??)
    - db: postgres
    - Password: example
    - I connected from DBeaver, you can see metrics here.
        - Good to try if Grafana doesn't work
- S3
    - <https://s3.console.aws.amazon.com/s3/buckets/sal-wine-quality?region=us-east-2&tab=properties>
    - The bucket is publicly accessbile
        - test.csv is the test data you can download to feed into the model (not used in train/valid of the model)
        - `1` contains all the MLFlow experiment runs
        - `models/BestWineDatasetModel/preds` contains the daily predictions
            - In this scenario, ground truth is available at scoring
            - Think about this as scoring a model once we attain ground truth data after the fact to see how we performed

********************************
Things I did not do:

- Terraform IaC
- CI/CD
- Prefect — Wanted to but ran out of time


********************************
********************************Next time:********************************
- I would chose a model with more of a time impact like we had with Taxis. I didn't foresee these issues, though I should've expected it. Given part of MLOps means how things move across time and operationalizing Machine Learning.

- I would explore more CI/CD and Terraform and Prefect.
    - I like Prefect but truthfully I haven't figured out how it plays into this project.

- More time!
    - WOW THIS WAS HARD
    - And I have so much more left to do
    - But I am so glad I did it.

********************************

Feedback:
- I want ALL the feedback. Please give me everything you see or can think of.

********************************
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
