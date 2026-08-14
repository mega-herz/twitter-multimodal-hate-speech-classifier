#import os
#import shutil
#from pathlib import Path
#import kagglehub

import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Dict
import zipfile



def load_data_from_json(
        json_path: Union[str, Path]
        ) -> pd.DataFrame:
    """
    Reads raw dataset JSON into a pandas DataFrame.

    Parameters:
        json_path (str or Path): 
            Full file path to the dataset JSON .

    Returns:
        pd.DataFrame: 
            Loaded DataFrame with 'file_id' as a column.
    """
    # Read raw data from JSON into a dictionary
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Load into DataFrame
    raw_df = pd.DataFrame.from_dict(json_data, orient='index')

    # Store tweet ID in dedicated column and reset index
    raw_df = raw_df.reset_index().rename(columns={'index': 'file_id'})

    return raw_df



def load_image_texts(text_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Reads OCR/text JSON files from a specified directory and maps each message ID 
    to its extracted text content.

    Parameters:
        text_dir (str or pathlib.Path): 
            Directory path containing the JSON files.

    Returns:
        dict: 
            Dictionary where keys are tweet IDs (file names without extension) 
            and values are the extracted text strings.
    """
    text_dir_path = Path(text_dir)
    text_data = {}

    # If no images exist
    if not text_dir_path.exists():
        return text_data

    # Iterate directly over all files in the directory
    for filename in os.listdir(text_dir_path):
        if not filename.endswith('.json'):
            continue

        # Extract message ID (e.g., '12345.json' -> '12345')
        msg_id = os.path.splitext(filename)[0]
        file_path = os.path.join(text_dir_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # Extract and join string values based on JSON structure
                if isinstance(content, dict):
                    text = " ".join([str(v) for v in content.values() if v])
                elif isinstance(content, list):
                    text = " ".join([str(item) for item in content if item])
                else:
                    text = str(content)

                text_data[msg_id] = text.strip()

        except (json.JSONDecodeError, OSError):
            # Skip unreadable or corrupted files
            continue

    return text_data



def load_split_ids(
    zip_path: str = 'multimodal-hate-speech.zip',
    split_dir: str = 'splits',
    as_int: bool = False,
) -> dict[str, set]:
    """
    Reads train, val, test subset IDs directly from a zipped archive without extracting.

    Arguments:
        zip_path : str, optional
            Path to the zip archive containing the split files. 
            Defaults to 'multimodal-hate-speech.zip'.
        split_dir : str, optional
            Directory path inside the zip archive where split ID files are stored. 
            Defaults to 'splits'.
        as_int : bool, optional
            Whether to convert the loaded IDs into integers. If False, IDs are kept as strings. 
            Defaults to False.

    Returns:
        dict[str, set]
            A dictionary mapping split names ('train', 'val', 'test') to sets of sample IDs.
    """
    splits = {}
    split_names = ['train', 'val', 'test']

    with zipfile.ZipFile(zip_path, 'r') as z:
        for split in split_names:
            file_path = f'{split_dir}/{split}_ids.txt'
            with z.open(file_path) as f:
                lines = (line.decode('utf-8').strip() for line in f)
                if as_int:
                    splits[split] = {
                        int(line) for line in lines if line.isdigit()
                    }
                else:
                    splits[split] = {line for line in lines if line}

    return splits