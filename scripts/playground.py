import requests
from tqdm import tqdm
import tarfile

base_url = 'https://veekun.com/static/pokedex/downloads'
url_list = ['generation-3.tar.gz', 'generation-4.tar.gz']
for url_ in url_list:
    url_zip = f'{base_url}/{url_}'
    response = requests.get(url_zip, stream=True)
    with open(f"./data/{url_}", "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

    tar = tarfile.open(f"./data/{url_}", "r:gz")
    tar.extractall()
    tar.close()