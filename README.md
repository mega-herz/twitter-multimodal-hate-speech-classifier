# Twitter Multimodal Hate Speech Classifier

## Description
A machine learning pipeline designed to detect and categorize multimodal (text and images) hate speech in tweets into a 6-class systematization (NotHate, Racist, Sexist, Homophobe, Religion, OtherHate). The project combines textual and visual data of tweeter messages and evaluates a variety of natural language processing techniques and machine learning algorithms to maximize classification performance.


## Task Formulation and Purpose

Automated content moderation on social media platforms is critical for minimizing online toxicity, protecting user safety, and scaling compliance efforts. This project develops a multiclass hate speech classifier for multimodal tweets (text and images), enabling to identify hateful content and prioritize its handling for rapid intervention.


## Data
**Dataset**: [MMHS150K - Multimodal Hate Speech Dataset](https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech) 

**Author**: Raúl Gómez Bruballa

**Description**: A large-scale, manually annotated dataset containing approximately 150 000 tweets. Each observation contains a raw tweet text with an accompanying image. The tweets are categorized into six distinct classes representing different forms of hateful content: NotHate, Racist, Sexist, Homophobe, Religion, and OtherHate. For images containing embedded text, OCR (Optical Character Recognition) was used for text extraction. The train, validation, and test splits provided by the original authors are included in the dataset.



## Evaluation Metrics
To evaluate model performance across highly imbalanced multiclass categories the following quality metrics were applied:
* **Primary Metric** - **Macro Average F1-Score**: Used to evaluate performance on all classes equally penalizing poor performance on minority classes.
* **Secondary Metric** - **Weighted Average F1-Score**: Used to evaluate overall performance considering frequency of each class in the dataset.



## Solution and Instruments

### Image Processing
**BLIP Model**: Used the Bootstrapping Language-Image Pre-training (BLIP) model to automatically generate descriptive text captions for images. This converts the visual content into text-based feature. The captions were generated for all pictures of the dataset, regardless of whether the images contained embedded text. These captions were used in further analysis along with other features.

### Models and Methods Explored
The text data was cleaned, preprocessed, and encoded to prepare it for the classification models. 

The following **text encoders** were evaluated in this project:

* **Bag of Words (BoW)**: A basic frequency-based representation that counts how often each word appears in a text, completely ignoring grammar, word order, and semantic context;

* **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical weighting technique that measures word importance by balancing its frequency within a specific document against its frequency across the entire text corpus, reducing the weight of common filler words.

* **TF-IDF with n-grams**: An extension of standard TF-IDF that tracks sequences of $n$ consecutive words (e.g. bigrams and trigrams) instead of relying purely on individual words alone, enabling the model to capture local context, phrasing, word order, and negations (such as "not hate").

* **Twitter-RoBERTa-base (cardiffnlp/twitter-roberta-base)**: A domain-adapted version of the Robustly Optimized BERT Approach (RoBERTa) pre-trained on 58 million tweets. Built on a state-of-the-art transformer architecture, it generates context-aware semantic embeddings that capture the nuanced meaning of words, slang, and informal syntax specific to social media.

The classification models were chosen to capture both linear and non-linear relationships within the data. The following combinations of text encoders and classification models were evaluated in this project:

1. **Bag of Words + Logistic Regression (baseline)**: A lightweight, interpretable performance baseline to measure the improvement of using more complex text representations and models;
2. **TF-IDF + XGBoost**: Combines statistical term-weighting (which highlights rare, informative words) with a powerful tree-based gradient-boosted ensemble model capable of capturing non-linear feature interactions;
3. **TF-IDF with n-grams + Linear SVC**: Captures local context and employs a robust linear classifier designed for high-dimensional, sparse feature matrices;
4. **Twitter-RoBERTa + XGBoost**: Combines deep, domain-adapted semantic embeddings from social media (Twitter) with a tree-based gradient-boosted ensemble model capable to identify non-linear and complex interactions within deep semantic features;
5. **Twitter-RoBERTa + Linear SVC**: Pairs domain-adapted semantic embeddings from social media (Twitter) with highly efficient, scalable exact linear classifier to test whether a simple model is sufficient when using deeply semantic features;
6. **Twitter-RoBERTa + SGDClassifier**: Pairs domain-adapted semantic embeddings from social media (Twitter) with scalable, stochastic linear classifier to test wether a simple model is sufficient when using deeply semantic features.

---

## Results and Conclusions

The results of the study are presented in the table below, showing the macro-averaged and the weighted-average F1-scores on the validation set.

| Model / Architecture | Feature Extraction Method | Macro Avg F1-Score (Primary) | Weighted Avg F1-Score (Secondary) |
| :--- | :--- | :---: | :---: |
| Baseline | Bag Of Words + Linear Regression | 0.5389 | 0.6208 |
| Linear Model | TF-IDF (ngram) + Linear SVC | 0.5507 | 0.6677 |
| Transformer Linear | RoBERTa + Linear SVC | 0.4927 | 0.6536 |
| Transformer SGD | RoBERTa + SGDClassifier | 0.4513 | 0.6341 |
| Transformer Boost | RoBERTa + XGBoost | 0.4875 | 0.5913 |
| **Best Model** | **TF-IDF + XGBoost** | 0.5827 | 0.6628 |
| **Tuned** | **TF-IDF + XGBoost** | 0.5866 | 0.6691 |

Overall, the approaches showed quite similar results. None of the appraoches demonstrated signs of overfitting.

An interesting pattern was observed in the performance of all approaches: their macro-averaged F1-scores on the validation data were slightly higher than on training data. Furthermore, the tuned version of best-performed approach, TF-IDF+XGBoost, showed slightly higher macro-averaged F1-score on test data compared to validation and training. Since the division on train, validation, and test subsets was pre-defined by the authors, this behavious likely stems from different imbalance of the classes in training, validation and test sets. 

TF-IDF + XGBoost showed slightly better performance during the experiemnts. Therefore, XGBoost was further tuned with Hyperopt which resulted in minor improvement in its F1-scores.  

---

## Repository Structure

```
├── data/                            # Raw data, splits (train, validation, test), preprocessed files     
├── notebooks/                       # Jupiter Notebooks detailing the project workflow
|   ├── 00_load_data.ipynb           # Loading and initial formatting of the dataset
|   ├── 01_generate_captions.ipynb   # Generation of descriptive captions for images attached to tweets.
|   ├── 02_EDA.ipynb                 # Exploratory Data Analysis
|   ├── 03_preprocessing.ipynb       # Text cleaning, normalization, and feature encoding pipelines.
|   ├── 04_models.ipynb              # Setup, training, and evaluation of the models
|   ├── 05_hyperparam_tuning.ipynb   # Hyperparameter optimization with Hyperopt for best-performing model
├── models/                          # Saved model artifacts (trained models, optimized configs, feature names)
├── reports/                         # Saved plots, performance summary table, confusion matrices
├── src/                             # Helper functions                   
├── README.md                        # Detailed project description
└── requirements.txt                 # Python package dependencies
```

## Installation and Usage

To run this project, we recommend to use Google Colab with Google Drive integration to ensure data persists across notebooks. 

### Prerequisites
Before getting started, make sure you have the following ready:

* A Google Account: With enough free space in your Google Drive to store the project repository and dataset.

* A Kaggle Account & API Key: Since the raw dataset is downloaded from Kaggle, you will need your Kaggle API token. 

* A Hugging Face Account and API Key: BLIP model is accessed through Hugging Face.

* A GPU Runtime (Recommended): Set your Colab runtime to T4 GPU (via Runtime > Change runtime type) for faster multimodal model training.

### Step 1: Open Notebook in Colab
Click the badge below to open the first pipeline notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mega_herz/twitter-multimodal-hate-speech-classifier/blob/main/00_load_data.ipynb)

### Step 2: Run the Setup Cell
1. To access the raw dataset, add and enable your Kaggle API token in the Secrets panel (key icon) of `00_load_data.ipynb`. 
2. At the very beginning of `00_load_data.ipynb`, run the initialization cell. It will:
    * Prompt you to connect your **Google Drive**.
    * **Automatically clone** this repository into your Google Drive (`/content/drive/MyDrive/twitter-multimodal-hate-speech-classifier`).
    * Create the shared `data/` directory.
    * Create `/content/local_dataset` directory in Colab runtime for further quick access to raw data
3. Follow the sequential steps in the notebook to download data.

### Step 3: Run the Pipeline
Execute the notebooks in sequential order to download and handle data and to run the models:
* **`00_load_data.ipynb`**: Downloads the raw dataset and saves it directly to Google Drive folder (`.../twitter-multimodal-hate-speech-classifier/data`);
* **`01_generate_captions.ipynb`**: Accesses the data from Google Drive, unpacks it, and runs the model to generate image captions. To access the BLIP model, add and enable your Hugging Face API token in the Secrets panel (key icon) of the notebook;
* **`02_EDA.ipynb`**: Performs minor data transformations and Exploratory Data Analysis;
* **`03_preprocessing.ipynb`**: Carries out cleaning and encoding of text features as well as scaling of numeric features;
* **`04_models.ipynb`**: Performs setting and training of the models as well as their evaluation;
* **`05_hyperparam_tuning.ipynb`**: Implements hyperparameter tuning for the best berforming model with Random Search and Hyperopt.