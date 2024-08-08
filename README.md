# (GigaTech) end-to-end NER & POS Tagging Classification.

[![Author](https://img.shields.io/badge/author-utshabkg-red)](https://github.com/utshabkg/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-blue.svg?style=flat)](https://github.com/utshabkg/end-to-end-nlp-practice/)
[![Stars](https://img.shields.io/github/stars/utshabkg/end-to-end-nlp-practice?style=social)](https://github.com/utshabkg/end-to-end-nlp-practice/stargazers)

## Prerequisites

- **Conda**: Ensure you have `Conda` installed on your system. If not, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). You can use `Virtualenv` or `Poetry` or any other tools too, if you know how to setup with them.

## Environment Setup

Follow these steps to set up your project environment:

1. **Clone the repository** (if you haven't done so already):

   ```bash
   git clone https://github.com/utshabkg/end-to-end-nlp-practice/
   cd end-to-end-nlp-practice
   ```

2. **Run the setup script**:

   This script will create a new Conda environment named `gigatech` with Python 3.10 and install all the required packages.

   ```bash
   bash setup_env.sh
   ```

3. **Activate the Conda environment**:

   After running the setup script, activate the new environment (if not activated):

   ```bash
   conda activate gigatech
   ```

4. **Verify the installation**:

   Ensure that all packages are installed correctly by running:

   ```bash
   pip list
   ```

   This should display a list of installed packages.

## Inference

You can Provide any Bangla Sentence and get the results. Available both in **Terminal** and a **Web Application** (powered by `FastAPI`)

### For Web Application

```bash
python main.py
```

Open your browser and go to http://localhost:8000/

### For Terminal

```bash
cd components
python inference.py
```

## Dockerize (Bonus)

```bash
docker build -t gigatech-app .    # build the image
docker run -d -p 8000:8000 fb8974ff6273    # run the container
docker ps    # check
```

Open your browser and go to http://localhost:8000/

NOTE: Please wait some time to load all the components after the running of container. You can see docker log with:
```bash
docker logs <container-id>
```
The container is ready to watch in the browser after this message appears in the log: 
```bash 
Application startup complete.
```

### Endpoint to check with Curl or Postman

Predict (Terminal should have Bangla Unicode Support to understand result):
```bash
curl -X POST "http://127.0.0.1:8000/predict_json" -H "Content-Type: application/x-www-form-urlencoded" -d "sentence=আমি বাংলা ভাষায় কথা বলি"
```
Health Check:

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**NOTE:** I have trained the model with `max_token=25`, so keep total number of words and punctuation within that. You can increase the token size and train a larger model too.

## Training Model with Custom Dataset & Evaluate

For creating a base model of your own to be preprocessed and trained with your data, run:

```bash
cd components
python preprocessing.py
python model_training.py
```

A model will be created at the path: `models/custom_data_model.h5`.

Evaluate your new model with:

```bash
cd components
python model_evaluation.py
```

You will get your results in: `reports/final_score_custom.txt` file.

**NOTE:** Data format should be the same as dataset folder. It should be a `.tsv` file. Rename your dataset file to `data.tsv`, keep it inside the `dataset` folder and you're all set!

## Data Exploration

If you are interested in Exploratory Data Analysis, Preprocessing, and other experiments (e.g. Hyperparameter Tuning) which I enjoyed, you can watch the `notebooks` folder.

## Documentation of Work Details
