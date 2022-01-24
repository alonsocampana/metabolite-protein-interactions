import os
import requests
import urllib
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import base64


class Error(Exception):
    """
    Base error class
    """
    pass


class ModelNotTrainedError(Error):
    """
    Model to be raised when the user tries to use an untrained model for inference.
    """
    pass


def progress_bar(current: int, total: int, length=50) -> str:
    """
    Returns a string containing a progress bar in function of the progress made.
    [current]: The iteration finished
    [total]: Total number of iterations
    """
    progress = "█"
    fill = "_"
    frac = int(current/total*length)
    bar = progress * frac + (length-frac) * fill + \
        "| " + '{0:.1f}'.format(current/total*100) + "%"
    return bar


def download_file_from_google_drive(id, destination):
    """
    source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """
    source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    """
    Adapted from source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        length = int(len(response.content)//CHUNK_SIZE) + 1
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            print(progress_bar(i+1, length), end="\r")
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        print(progress_bar(1, 1), end="\r")
        print("")
