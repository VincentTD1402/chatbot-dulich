import os
import random
import time
import numpy as np
import openai
import streamlit as st
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from utils import read_file, sentence_to_generator, show_time_sleep_generator, show_async_generator
from create_chromadb import LocalChromaDB
from temp_chroma_db import TempChromaDB
from templates.prompt import openai_for_answer
from config import config
from templates.prompt import openai_for_answer_with_context


client = OpenAI(api_key="sk-proj-ji23rzuXo3gLwEsZzaTYT3BlbkFJOP5ha5Guh6Ke8zM8rlbY")

list_location_path = "data/diadiemdulich.txt"
list_location = read_file(list_location_path)

chroma_db = LocalChromaDB()
temp_chroma_db = TempChromaDB()
custom_css = f"""
<style>
    .st-emotion-cache-16idsys p {{
        font-size: 20px; /* You can adjust the font size as needed */
        font-weight: bold;
    }}

    .st-emotion-cache-1v0mbdj img {{
        
    }}
</style>
"""

# State management
CURRENT_PAGE = "main"

def question_search(name_diadiem: str,
        question: str,
        num_of_answers = 1):
    '''
    Search the similarity queries in the database
    Args:
        - question (str): The question of stories
        - num_of_answers (int): The number of answer for stories
    Returns:
        A list of similar questions
    '''
    dulich_collection = client.get_collection(
        name='dulich_simcse',
        embedding_function=sentence_transformer_ef)
    results = dulich_collection.query(
        query_texts=[question],
        n_results=num_of_answers,
        where_document={"$contains": name_diadiem}
    )
    for i in results['documents'][0]:
        print('Answer: ', i)
    return results['documents']

def query_make(question_input: str, localion_selected: str):
    '''
    Querying the question based on the story selected
    Args:
        - question_input (str): question input
        - story_selected (str): story is selected
    Returns:
        a full sentence based on template
    '''
    query_sen = "ở" + localion_selected + ", " + question_input
    return query_sen

def save_data(prompt, data_accept):
    '''
    Saving the data
    Args:
        - prompt (str): Prompting data to save
        - data_accept (str): data_accept
    '''
    return 0

async def main():
    location_selected = st.sidebar.selectbox(
        "🌟 Location Selection",
        list_location
    )
    st.markdown(custom_css, unsafe_allow_html=True)
    if location_selected != "":
        st.image(
            "image/chatbot.jpg",
            width=None,  # Manually Adjust the width of the image as per requirement
        )
    st.title("💬 Vietnam Travel Chatbot")
    if "messages" not in st.session_state:
        FIRST_ASSISTANT_MESSAGE = "Xin Chào, tôi là trợ lý tư vấn du lịch cá nhân, tôi có thể giúp gì cho bạn?"
        st.session_state["messages"] = [{"role": "assistant", "content": FIRST_ASSISTANT_MESSAGE}] 
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)

    answer = None
    full_response = ''
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        sim_anwer = chroma_db.find_sim_answer(
            name_collection='dulich_simcse',
            name_diadiem=location_selected,
            question=prompt,
            num_of_answer=1
        )
        answer=sim_anwer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            start_time = datetime.now()  # Record start time
            if answer:
                generator = sentence_to_generator(answer)
                message_placeholder, full_response = \
                    show_time_sleep_generator(message_placeholder, generator)
            else:
                st.markdown(f"🔎 searching external source")
                external_chunks = await temp_chroma_db.find_external_chunks(prompt)
                new_prompt = openai_for_answer_with_context(
                    question=prompt,
                    context=external_chunks['context']
                )
                print("new prompt",new_prompt)
                # Update the last message by adding context into prompt.
                update_messages = st.session_state.messages.copy()
                update_messages[-1] = {"role":"user", "content":new_prompt}
                # Request chatbot
                # openai.api_key = openai_api_key
                st.markdown(f"🚀 generating content")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0613", 
                    temperature=0.01,
                    stream=True,
                    messages=update_messages)
                print("response:",response)
                for token in response:
                    # print(token)
                    try:
                        # value = token['choices'][0]['delta']['content']
                        value = token.choices[0].delta.content
                        # print(value)
                        full_response += value
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    except Exception as e:
                        print(e)
                        pass
            if st.button("Response"):
                CURRENT_PAGE = "response"
                response_page()

                
            end_time = datetime.now()
            response_time = end_time - start_time
            role_of_last_anwser = st.session_state["messages"][-1]['role']
            if role_of_last_anwser == "user":
                st.markdown(f"🕒 Bot response time: {response_time.total_seconds()} seconds")
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})   

def response_page():
    st.title("Response Page")
    user_input = st.text_input("Enter your response:")
    if user_input:
        st.write(f"You entered: {user_input}")
        if st.button("Back to Main"):
            CURRENT_PAGE = "main"

async def start_app():
    # Determine which page to display (e.g., CURRENT_PAGE)
    if CURRENT_PAGE == 'response':
        response_page()
    else:
        await main()

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    asyncio.run(start_app())