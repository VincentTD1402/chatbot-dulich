import os
import re
import requests
import json
from bs4 import BeautifulSoup
import argparse
import trafilatura
import httpx


class WebPageTextExtractor(object):
    """
    A class to extract text from a web page.
    ------------
    Attributes:
        url: (str) -> The URL of the web page.
        div_class: (str) -> The class of the <div> tag containing the text.
        parenthesis_regex: (object) -> A regular expression object to remove parenthesis content.
        citations_regex: (object) -> A regular expression object to remove citations, e.g., [1].
    """
    def __init__(self, url: str) -> None:
        self.url = url
        self.div_class = None
        self.parenthesis_regex = re.compile(r'\(.+?\)')  # To remove parenthesis content
        self.citations_regex = re.compile(r'\[.+?\]')    # To remove citations, e.g., [1]
        self.NEWS_CLASS_MAPPING = {
            'vnexpress': 'sidebar-1',
            'cafef': 'left_cate totalcontentdetail',
            'vietstock': 'content',
            'wikipedia': 'mw-content-container',
            'tinnhanhchungkhoan': 'leftBlock wrap_noi_dung',
            'thanhnien': 'detail__cmain',
            'mof.gov': 'new-content cd-content',
            'vneconomy.vn': 'detail__header',
            'nhandan.vn': 'main-content article',
            'baochinhphu.vn': 'detail-mcontent',
            'tapchicongthuong.vn': 'post-content',
            'tapchitaichinh.vn': 'detail-wrap',
            'quochoi.vn': 'container',
            'vtv.vn': 'noidung',
            'www.tapchicongsan.org.vn': 'clearfix ContentDetail',
            'baodautu.vn': 'col630 ml-auto mb40',
            'tuoitre.vn': 'detail__cmain',
            'laodong.vn': 'pl',
            'dangcongsan.vn': 'detail-post hnoneview',
            'kinhtevadubao.vn': 'post',
            'tapchinganhang.gov.vn': 'col_left',
            'dantri.com.vn': 'singular-wrap',
            'vietjack.com': 'col-md-7 middle-col',
            'loigiaihay.com': 'box-question top20',
            'vanhocthpt.com': 'post-content',
            'thichvanhoc.com.vn': 'entry-content single-page',
            'vndoc.com': 'maincontent textview padingads',
            'download.vn': 'textview content-detail',
            'luatminhkhue.vn': 'hurray post-content entry-content',
            'hoatieu.vn': 'maincontent textview',
            'thuthuat.taimienphi.vn': 'des3 clearfix',
            'cunghocvui.com': 'card card-post-detail my-0',
            'toploigiai.vn':'row category-page-border lesson-detail',
            'haylamdo.com': 'main-content-detail',
            'butbi.hocmai.vn': 'td-post-content',
            'webhoctot.com': 'single_page',
            'lopvancothu.com':'entry-content',
            'hoc247.net': 'itvc20player'
        }
        for key, value in self.NEWS_CLASS_MAPPING.items():
            if key in self.url:
                self.div_class = value


    def get_text_from_tag(self, tag: BeautifulSoup) -> str:
        """
        A recursive function to extract text from header and content tags within a <div> tag.
        The function will recursively call itself to extract text from child tags.
        It will ignore tags that contain redundant information, e.g., cite, symbol, etc.
        -----------
        Args:
            tag: (object) -> A BeautifulSoup tag object.
        Returns:
            extract_text: (str) -> A string containing the extracted text.
        """
        extracted_text = ""
        # Check if the tag is a header tag (h1 to h6) or a content tag (p)
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6", "p"]:
            if not tag.find_all("span"):
                text = tag.get_text(strip=True)
                if text:
                    text = self.parenthesis_regex.sub('', text)
                    text = self.citations_regex.sub('', text)
                    extracted_text += text + "\n"

        for child_tag in tag.children:
            if isinstance(child_tag, str):
                continue
            extracted_text += self.get_text_from_tag(child_tag)

        return extracted_text


    async def get_text_from_div(self) -> str:
        """
        A function to get text from header and content tags within a <div> tag.
        ------------
        Returns:
            extracted_text: (str) -> A string containing the extracted text.
        """
        try:
            async with httpx.AsyncClient() as client:
                print('throw a request')
                response = await client.get(self.url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    div_tag = soup.find('div', class_=self.div_class)
                    if div_tag:
                        extracted_text = self.get_text_from_tag(div_tag)
                        if extracted_text != "":
                            return extracted_text
                        else:
                            trafilatura_text = trafilatura.extract(response.text)
                            trafilatura_text = self.parenthesis_regex.sub('', trafilatura_text)
                            trafilatura_text = self.citations_regex.sub('', trafilatura_text)
                            return trafilatura_text
                    else:
                        trafilatura_text = trafilatura.extract(response.text)
                        trafilatura_text = self.parenthesis_regex.sub('', trafilatura_text)
                        trafilatura_text = self.citations_regex.sub('', trafilatura_text)
                        return trafilatura_text
                else:
                    response = trafilatura.fetch_url(self.url)
                    trafilatura_text = trafilatura.extract(
                                            response, 
                                            include_comments=False, 
                                            include_tables=False, 
                                            no_fallback=True, 
                                            include_links=False, 
                                            include_images=False, 
                                            deduplicate=False
                                        )
                    trafilatura_text = self.parenthesis_regex.sub('', trafilatura_text)
                    trafilatura_text = self.citations_regex.sub('', trafilatura_text)
                    return trafilatura_text
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None


    def save_text_to_file(self, output_dir:str, file_name: str) -> None:
        """
        A function to save the extracted text to a file.
        ------------
        Args:
            output_dir: (str) -> The output directory to save the text file.
            file_name: (str) -> The name of the output text file.
            
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                text = self.get_text_from_div()
                if text:
                    file.write(text)
                else:
                    print("No text to save.")
        except Exception as e:
            print(f"An error occurred while saving the file: {str(e)}")


    def get_output(self) -> str:
        """
        A function to get the extracted text.
        ------------
        Returns:
            extracted_text: (str) -> A string containing the extracted text.
        """
        try:
            extracted_text = self.get_text_from_div()
            if extracted_text:
                return extracted_text
            else:
                return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Web Page Text Extractor")
    parser.add_argument("url", type=str, help="URL of the web page to extract text from")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory to save the text file")
    parser.add_argument("--file-name", type=str, default="output.txt", help="Name of the output text file")

    args = parser.parse_args()
    url = args.url
    output_dir = args.output_dir
    file_name = args.file_name

    text_extractor = WebPageTextExtractor(url)
    text_extractor.save_text_to_file(output_dir, file_name)
    print(f"Text saved to: {os.path.join(output_dir, file_name)}")


if __name__ == "__main__":
    # main()
    url = r'https://hoc247.net/ngu-van-12/khai-quat-van-hoc-viet-nam-tu-dau-cach-mang-thang-tam-1945-den-the-ki-xx-l3418.html'
    text_extractor = WebPageTextExtractor(url)
    text = text_extractor.get_output()
    print(text)
    # print(news_class_mapping)