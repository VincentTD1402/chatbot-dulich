import time
import asyncio
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googlesearch import search
from sklearn.feature_extraction.text import TfidfVectorizer
from config import config
from external_search.user_query import get_data

class TempChromaDB():
    client = PersistentClient(
        path='temp'
    )
    model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    tf_idf = TfidfVectorizer()
    temp_collection=None

    def __init__(self):
        pass

    def check_db(self):
        """
        Check whether database exists
        """
        if self.client.list_collections():
            print('client exist!')
            return True
        else:
            print("client doesn't exist")
            return False
    

    def split_context(self, 
        query: str, 
        documents: list[dict]) -> list[str]:
        """
        Split documents into chunks and find a list of top similarity with a query
        Args:
            query(str:None): question need to be answered
            documents(list[dict]:None): list of documents, each document includes content and url
        Return:
            list of chunks (list[str])
        """
        start = time.time()
        chunks = []
        meta_datas = []

        for d in documents:
            document = d['content']
            url = d['url']
            text_splits = self.splitter.split_text(document)
            chunks += text_splits
            text_splits = [(item, url) for item in text_splits]
            meta_datas += text_splits

        if len(chunks) <= 10:
            print(f"Splitting in: {time.time()-start}")
            return chunks

        self.tf_idf.fit([query+'. '] + chunks)
        embedded_query = self.tf_idf.transform([query])

        similarity = []
        for chunk in chunks:
            similarity.append(
                cosine_similarity(self.tf_idf.transform([chunk]), embedded_query)[0][0]
            )
        meta_datas = [{'content': item[0], 'url': item[1]} for (i, item) in enumerate(meta_datas) if i in np.array(similarity).argsort()[-10:]]
        
        print(f"Splitting in: {time.time()-start}")
        return meta_datas

    def create_collection(self, 
            resp: list[dict],
            name_collection: str='temp'
        ):
        """
        Create a temporary collection in database for Vietnam Literary Story
        Args:
            - resp (list[dict]): a list of dictionary to save to a collection. 
            Each dictionary include {'content':, 'url':}.
            - name_collection (str='temp'): a name of collection.
        """
        start = time.time()
        try:
            self.client.delete_collection(name_collection)
            print(f"All samples of '{name_collection}' has been removed.")
        except Exception:
            pass

        self.temp_collection = self.client.get_or_create_collection(
            name=name_collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.sentence_transformer_ef
        )
        print(f"Create a new collection '{name_collection}' successfully")
    
        documents, embeddings, meta_datas, ids= [], [], [], []
        for index, doc in enumerate(resp):
            content = doc['content']
            url = doc['url']
            documents.append(content)
            embedding = self.model.encode(content).tolist()
            embeddings.append(embedding)
            meta_datas.append({'source':url})
            ids.append(str(index+1))
        
        self.temp_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=meta_datas,
            ids=ids
        )

        print(f"Create collection temp in chromadb: {time.time() - start}")

    async def find_external_chunks(self, query: str):
        """
        Return a list of chunks from a customer's query
        Args:
            - query (str): the query of customer
        """
        # Step 1: clone data from internet
        download_content = await get_data(query, num_urls=5)
        # Step 2: split downloaded content into chunks, then filtering chunks by TF-IDF to get top 10 similar chunks
        meta_datas = self.split_context(query, download_content)
        # Step 3: create a collection of meta_datas
        self.create_collection(meta_datas)
        # Step 4: query the most similar context
        result = self.temp_collection.query(
            query_texts=[query],
            n_results=1
        )
        context = result['documents'][0][0]
        url = result['metadatas'][0][0]['source']
        return {'query': query, 'context': context, 'url': url}

# if __name__ == "__main__":
#     # 1. Test creating chromadb
#     chromadb = TempChromaDB()
#     # query = 'Tóm tắt truyện ngắn Lão Hạc'
#     # result = chromadb.find_external_chunks(query)
#     # print(result)