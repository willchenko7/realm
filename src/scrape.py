'''
goal: given a url, scrape the page for the text on it
'''

import requests
from bs4 import BeautifulSoup
from csv_helpers import txt2csv

def scrape(url):
    # Fetch the webpage content
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all text and remove extra whitespaces
    return ' '.join(soup.stripped_strings)

def multiple_scrapes(urls):
    text = ""
    for url in urls:
        new_text = scrape(url)
        text += " "
        text += new_text
    return text

fn = 'beyond_good_and_evil.csv'
urls = [
    "https://archive.org/stream/NietzscheBeyondGoodAndEvil/Nietzsche-Beyond-Good-and-Evil_djvu.txt"
    ]
text = multiple_scrapes(urls)
print(text)
txt2csv(text,fn)