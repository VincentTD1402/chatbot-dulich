from googlesearch import search
import asyncio
import time
from .web_crawler import WebPageTextExtractor
list_tails = ['.html', '.htm', '.chn', '.aspx', '.ldo']

def get_urls(query, num_urls):
    """
    A function to get a list of URLs from Google search results.
    Args:
        query: (str) -> The query string.
        num_urls: (int) -> The number of URLs to return.
    Returns:
        urls: (list) -> A list of URLs.
    """
    urls = []
    # for url in search(query, tld="co.in", num=10, stop=num_urls, pause=2):
    #     for val in list_tails:
    #         if val in url and 'dfat.gov' not in url:
    #             urls.append(url)
    urls = search(query, tld="co.in", num=5, stop=num_urls, pause=2)
    return urls
    
# def delete_all_file(path):
#     files = os.listdir(path)
    
#     for file in files:
#         f = os.path.join(path,file)
#         if os.path.isfile(f):
#             os.remove(f)
    
# def get_data(query,num_urls = 2,query_folder = 'data'):
#     def process_url(url, i):
#         file_name = f'_{i}.txt'
#         print(url,file_name)
#         run = 'python src/getdata/web_extractor.py {0} --output-dir={1} --file-name={2}'.format(url, query_folder, file_name)
#         os.system(run)

async def fetch_and_process_url(url):
    start = time.time()
    text_extractor = WebPageTextExtractor(url)
    text = await text_extractor.get_text_from_div()
    print('executing time for one request {}'.format(time.time() - start))
    return "{}###{}".format(text, url)

async def get_data(query, num_urls = 5) -> list[dict]:
    start = time.time()
    urls = get_urls(query, num_urls)
    urls = set(urls)
    tasks = [fetch_and_process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    parsed_results = []
    for item in results:
        (content, url) = item.split("###")
        parsed_results.append({'content': content, 'url': url})
    print('crawl in:', time.time() - start)
    return parsed_results