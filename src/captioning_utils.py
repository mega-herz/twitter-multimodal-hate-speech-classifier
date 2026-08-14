import os
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
import zipfile
from IPython.display import display
from PIL import Image



def load_blip_model(
    model_id: str = "Salesforce/blip-image-captioning-base",
    device: str | None = None,
) -> tuple[BlipProcessor, BlipForConditionalGeneration, str]:
    """
    Loads BLIP processor and model to the target device.

    Parameters:
        model_id (str):
            Hugging Face model repository ID or local path for the BLIP model.
        device (str | None): 
            Target device to load the model onto ('cuda' or 'cpu'). 
            If None, automatically selects CUDA if available, otherwise falls back to CPU.

    Returns:
        tuple[BlipProcessor, BlipForConditionalGeneration, str]: A tuple containing:
            - The initialized BLIP processor.
            - The BLIP model loaded onto the specified device.
            - The string identifier of the device used ('cuda' or 'cpu').
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

    return processor, model, device



def generate_batch_description(
    image_paths: list[str],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: str,
) -> list[str]:
    """
    Loads a batch of image files and generates BLIP captions.

    Parameters:
        image_paths (list[str]): 
            List of file paths pointing to the images to be processed.
        processor (BlipProcessor): 
            BLIP processor used to preprocess and tokenize the images.
        model (BlipForConditionalGeneration): 
            BLIP model used for conditional text generation.
        device (str): 
            Target device the model is loaded on ('cuda' or 'cpu').

    Returns:
        list[str]: 
            A list of generated captions matching the order of input paths. 
            Corrupt or missing images return an empty string ("") at their respective index.
    """
    images = []
    valid_indices = []

    # Load images 
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(idx)
        except Exception:
            # Handle corrupt/missing images safely
            images.append(None)
            print(f'Image {path} doesnt exist.')

    captions = [""] * len(image_paths)

    # Filter out None values before sending to GPU
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        print(f'No valid images found.')
        return captions

    # Process batch on GPU
    inputs = processor(
        valid_images,
        text=[""] * len(valid_images),
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)

    decoded_captions = processor.batch_decode(outputs, skip_special_tokens=True)

    # Map decoded captions back to their indices
    for idx, caption in zip(valid_indices, decoded_captions):
        captions[idx] = caption

    return captions



def display_image(
        zip_path: str, 
        image_dir_in_zip: str, 
        image_name: str
) -> None:
    """
    Loads an image from a zip file and displays it in the notebook output.

    Parameters:
        zip_path (str): 
            File path to the required .zip archive.
        image_dir_in_zip (str): 
            Directory path inside the zip archive where the image is located.
        image_name (str): 
            Name of the image file (without the extension).

    Returns:
        None
    """
    image_path = os.path.join(image_dir_in_zip, image_name + ".jpg")
    with zipfile.ZipFile(zip_path, "r") as archive:
        # Open the file inside the zip as a binary stream
        with archive.open(image_path) as file_stream:
            # Load into PIL Image
            img = Image.open(file_stream)
            
            # Display the image in the notebook output
            display(img)