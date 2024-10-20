# end-to-end (Bengali) NER & POS Tagging Classification.

[![Author](https://img.shields.io/badge/author-utshabkg-red)](https://github.com/utshabkg/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-blue.svg?style=flat)](https://github.com/utshabkg/ner-pos-tagging/)
[![Stars](https://img.shields.io/github/stars/utshabkg/ner-pos-tagging?style=social)](https://github.com/utshabkg/ner-pos-tagging/stargazers)

## Prerequisites

- For testing the deployed project in the cloud, there is no prerequisite! Go to [this link](https://ner-pos-tagging.onrender.com/) and test with any Bangla sentence you want.
  
  **NOTE:** Since it's a free instance of Render, it will spin down with inactivity, which can delay requests by 50 seconds or more. **So, please wait for 1-2 minutes to initiate the web app.** For now, the Bangla sentence may have a maximum of 25 tokens as the base model was trained with that size.
  A quick look at the web application:
  
  ![web_app](reports/web_app.jpg)
- Now it's time to run and test the project in your system. For that:

  **Conda**: Ensure your system has `Conda` installed. If not, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). You can use `Virtualenv` or `Poetry` or any other tools too, if you know how to set up with them.

## Environment Setup

Follow these steps to set up your project environment:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/utshabkg/ner-pos-tagging/
   cd ner-pos-tagging
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

### Web Application

```bash
python main.py
```

Open your browser and go to http://localhost:8000/

### For Terminal

```bash
cd components
python inference.py
```

### Dockerize (Bonus)

```bash
docker build -t gigatech-app .    # build the image
docker run -d -p 8000:8000 gigatech-app    # run the container
docker ps    # check
docker stop <container-id-from-ps>    # stop
```

Open your browser and go to http://localhost:8000/

**NOTE:** Please wait some time to load all the components after running the container. You can see the docker log with:

```bash
docker logs <container-id-from-ps>
```

The container is ready to watch in the browser after this message appears in the log:

```bash
Application startup complete.
```

### Endpoints to check with Curl or Postman

**Predict**: (Terminal should have Bangla Unicode Support to understand result)

```bash
curl -X POST "http://127.0.0.1:8000/predict_json" -H "Content-Type: application/x-www-form-urlencoded" -d "sentence=আমি বাংলা ভাষায় কথা বলি"
```

**Health Check:**

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**NOTE:** I have trained the model with `max_token=25`, so keep the total number of words and punctuation within that. You can increase the token size and train a larger model too.

### ONNX Integration

```bash
cd components/utils
python convert_model_onnx.py    # convert model to onnx
```

A model (has been already) created at the path: `notebooks/models_evaluation/models/base_model.onnx`.

**Inference**

```bash
cd components
python inference_onnx.py
```

## Training Model with Custom Dataset & Evaluate

For creating a base model of your own to be preprocessed and trained with your data, run:

```bash
cd components
python preprocessing.py
python model_training.py
```

A model will be created at the path: `notebooks/models_evaluation/models/custom_data_model.h5`.

Evaluate your new model with:

```bash
cd components
python model_evaluation.py
```

You will get your results in: `reports/final_score_custom.txt` file.

**NOTE:** Data format should be the same as the dataset folder. It should be a `.tsv` file. Rename your dataset file to `data.tsv`, keep it inside the `dataset` folder and you're all set!

## Data Exploration

If you are interested in Exploratory Data Analysis, Preprocessing, and other experiments (e.g. Hyperparameter Tuning) which I enjoyed, you can watch the `notebooks` folder.

## Documentation of Work Details

A document explaining the code and decisions made during the development process. [Click here](https://github.com/utshabkg/ner-pos-tagging/blob/main/EXPLANATION.md).

## Performance Metrics

A report of the model's performance on the test set, including accuracy, precision, recall, and F1 score. [Click here](https://github.com/utshabkg/ner-pos-tagging/blob/main/reports/score_1_base.txt).

### Base Model

A Plotting of training and validation accuracy and loss plot during a [base model](https://github.com/utshabkg/ner-pos-tagging/blob/main/notebooks/models_evaluation/models/base_model.h5) training.

![accuracy_plot](reports/accuracy_plot.png)
![loss_plot](reports/loss_plot.png)

**Hyperparameter tuning** was executed too. If you want, you can explore the [`models/parameters_track`](https://github.com/utshabkg/ner-pos-tagging/tree/main/notebooks/models_evaluation/models/parameters_track) folder to see the outcomes.
