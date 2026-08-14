from typing import Union, Any, List, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import re
import emoji
from emoticon_fix import emoticon_fix
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import torch
from tqdm import tqdm


# Module-level default instance
_DEFAULT_STEMMER = SnowballStemmer("english")


def extract_timestamp_features(
    timestamp_column: Union[pd.Series, pd.DataFrame, np.ndarray, list]
) -> npt.NDArray[np.float64]:
    """
    Extract cyclical time, weekend indicator, and month features from timestamps.

    Converts the input timestamp sequence into a 2D NumPy array containing:
    1. Sine-encoded hour of day (24-hour cycle)
    2. Cosine-encoded hour of day (24-hour cycle)
    3. Weekend flag (1 for Saturday/Sunday, 0 for weekdays)
    4. Month of the year (1 to 12)

    Parameters:
       timestamp_column : Union[pd.Series, pd.DataFrame, np.ndarray, list]
            A 1D or single-column 2D sequence containing datetime-like objects or strings.

    Returns:
        npt.NDArray[np.float64]
            A 2D NumPy array of shape (N, 4) containing the engineered timestamp features
            ordered as [hour_sin, hour_cos, is_weekend, month].
    """
    # Ensure input is a Series of datetimes
    dt_series = pd.to_datetime(pd.Series(timestamp_column.squeeze()))

    # Time-of-day cyclical features (24-hour)
    hour = dt_series.dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # Weekend indicator
    is_weekend = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

    # Month
    month = dt_series.dt.month

    # Return 2D array
    return np.column_stack([hour_sin, hour_cos, is_weekend, month])




def combine_text_columns(
    X: Union[pd.DataFrame, np.ndarray, list, dict]
) -> pd.Series:
    """
    Combine multiple text columns into a single 1D pandas Series.

    Fills missing values (NaNs), if there are any, with empty strings '' and concatenates the text 
    across columns for each row, separated by spaces.

    Parameters:
    X : Union[pd.DataFrame, np.ndarray, list, dict]
        A 2D or 1D tabular dataset (DataFrame, NumPy array, dictionary, or list) 
        containing text columns to be concatenated.

    Returns:
    pd.Series
        A 1D pandas Series of strings where each element represents the 
        space-separated concatenation of all input text columns for that row.
    """
    # Ensure X is a DataFrame and fill missing values with empty strings ''
    df = pd.DataFrame(X).fillna('')

    # Join selected columns row-by-row with space separator
    return df.agg(' '.join, axis=1)




def text_preprocessor(text: Any) -> str:
    """
    Clean raw text for NLP tasks.

    Performs several preprocessing steps including URL removal, emoticon and 
    emoji text conversion, username handle stripping, lowercasing, underscore 
    replacement, and character repetition reduction.

    Parameters:
    text : Any
        Input text string to process. If non-string values or NaNs are 
        passed, the function handles them by returning an empty string.

    Returns:
    str
        The cleaned, normalized plain text string.
    """

    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Handle emoticons and emojis: replaces it with the word
    text = emoticon_fix(text)
    text = emoji.demojize(text)

    # Remove @usernames (cleans @ followed by charactes until space)
    text = re.sub(r"@\S+", "", text)

    # Lowercase
    text = text.lower()

    # Turn underscores into spaces (__ -> ' ', _aston -> ' aston')
    text = text.replace("_", " ")

    # Reduce 3+ repeating characters to 2 (e.g., "loooove" -> "loove")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text


def text_tokenizer(
    text: str, 
    stemmer: Optional[SnowballStemmer] = None
) -> List[str]:
    """
    Tokenize text into words and apply stemming.

    Parameters:
    text : str
        The input string to tokenize and stem.
    stemmer : Optional[SnowballStemmer], default=None
        An NLTK stemmer instance. If None, defaults to an English SnowballStemmer.

    Returns:
    List[str]
        A list of stemmed token strings.
    """
    if not isinstance(text, str):
        return []

    if stemmer is None:
        stemmer = _DEFAULT_STEMMER

    raw_tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in raw_tokens]

    return stemmed_tokens




def twitter_roberta_preprocessor(text: str) -> str:
    """
    Replaces @usernames with '@user' and links with 'http' per CardiffNLP standard.
    
    Parameters:
        text (str): 
            The input text to be preprocessed.
    
    Returns:
        str: 
            Preprocessed text.
    """
    
    if not isinstance(text, str):
        text = str(text)

    new_text = []

    for token in text.split(" "):
        token = '@user' if token.startswith('@') and len(token) > 1 else token
        token = 'http' if token.startswith('http') else token
        new_text.append(token)

    return " ".join(new_text)


def extract_roberta_embeddings(
    X, model, tokenizer, device, batch_size=32, max_length=128
) -> np.ndarray:
    """
    Takes text input (Series, array, or list) and generates mean-pooled RoBERTa embeddings.

    Parameters:
        X (pd.Series, np.ndarray, or list): 
            Text data to encode.
        model (torch.nn.Module): 
            Pre-trained RoBERTa model.
        tokenizer (PreTrainedTokenizer): 
            Tokenizer corresponding to RoBERTa model.
        device (torch.device or str): 
            Hardware device ('cuda' or 'cpu') to run on.
        batch_size (int, optional): 
            Number of texts to process in each batch. Default is 32.
        max_length (int, optional): 
            Maximum sequence length for tokenization. Default is 128.

    Returns:
        np.ndarray: 
            2D NumPy array containing the generated mean-pooled embeddings of shape (n_samples, embedding_dimension).
    """
    if isinstance(X, pd.Series):
        texts = X.tolist()
    elif isinstance(X, np.ndarray):
        texts = X.ravel().tolist()
    else:
        texts = list(X)

    # Apply CardiffNLP preprocessing
    texts = [twitter_roberta_preprocessor(t) for t in texts]

    embeddings = []
    #model.eval()  # Ensure model is in evaluation mode

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)

            # Mean Pooling over hidden states
            mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(mean_pooled)

    return np.vstack(embeddings)


