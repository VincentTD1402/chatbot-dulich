import os
import asyncio
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.docstore.document import Document
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = "sk-proj-ji23rzuXo3gLwEsZzaTYT3BlbkFJOP5ha5Guh6Ke8zM8rlbY"
openai.api_key = os.environ["OPENAI_API_KEY"]


class HandleQA:
    def __init__(self, config):
        self.config = config
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding,
            cache_folder="cache",
            # model_kwargs={"device": "cpu"},
        )
        self.chroma = None
        self.callback = AsyncIteratorCallbackHandler()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",
            temperature=0.1,
            verbose=False,
            streaming=True,
            callbacks=[self.callback],
        )
    
        self.memory = ConversationTokenBufferMemory(llm=self.llm,max_token_limit=1500)
        self.chain = ConversationChain(llm=self.llm, 
                                       memory=self.memory, 
                                       verbose=True,
                                       )


    def openai_for_answer(self, question, answer):
        prompt = """
        Bạn là một hướng dẫn viên du lịch và được yêu cầu viết lại câu trả lời của câu hỏi dưới đây mà vẫn giữ nguyên ý nghĩa, văn phong và độ chính xác của câu trả lời theo cách một hướng dẫn viên du lịch. Phần trả lời được viết lại ở câu trả lời viết lại.
        Câu hỏi: {}
        Câu trả lời: {}
        Câu trả lời viết lại:
        """.format(question, answer)
        completion = client.chat.completions.create.create(
            model="gpt-3.5-turbo-0613", 
            messages=[{"role": "user", "content": prompt}])
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def reset_callback(self):
        self.callback.done.clear()

    async def ask_gpt(self, query: str, answer: str):
        """
        Return the answer from the chatGPT
        Args:
            query (str:None): The query that the user asked the chatbot
            answer (str:None): The answer from RAG pipeline
        Return:
            answer (str)
        """
        message = self.openai_for_answer(query, answer)
        task = asyncio.create_task(
            self.chain.arun(input=message)
        )
        try:
            async for token in self.callback.aiter():
                yield token
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            self.callback.done.set()
        await task