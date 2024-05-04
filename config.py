from pydantic_settings import BaseSettings
class Config(BaseSettings):
    embedding:str
    chunk_size: int
    chunk_overlap: int

config = Config(
    embedding = "keepitreal/vietnamese-sbert",
    chunk_size=500,
    chunk_overlap=0
)

class ConfigChat(BaseSettings):
    chunk_size: int
    chunk_overlap: int
    embedding: str
    similarity_function: str
    number_of_chunk: int


config_chat = ConfigChat(
    chunk_size = 2000,
    chunk_overlap=0,
    embedding="keepitreal/vietnamese-sbert",
    similarity_function="max_marginal_relevance_search",
    number_of_chunk=3
)