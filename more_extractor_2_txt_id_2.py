import os
import re
import openai
import streamlit as st
import csv
from datetime import datetime as dt
import pandas as pd
#from numpy import mean
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

#modeled on Baconbot_1_6_2.py build

st.set_page_config(
    page_title='Quote Extractor',
    layout='wide',
    page_icon='🔍'
)

st.title('Quote Extractor: "The History of Richard III" by Thomas More')
col1, col2 = st.columns([3.0,3.5])
with col1:
    book_pic = st.image(image ='./more_page.jpg', caption="From Thomas More's 'History of Richard III' in the British Library (1557). British Library.", width=500)
    #st.write("Explore the current data.")
    #df = pd.read_csv('richardbot1_data.csv')
    #st.dataframe(df, height=500)


def button_one():
    st.write("I am an AI research assistant. You can ask me questions about Thomas More's [_History of King Richard III_](https://thomasmorestudies.org/wp-content/uploads/2020/09/Richard.pdf) and I will answer with quotes taken from the text.")
    temperature_dial = st.slider("Temperature Dial", 0.0, 1.0)
    response_length = st.slider("Response Length", 1, 1000)
    submission_text = st.text_area("Enter your questions below. Be patient as I consider your inquiry.")
    submit_button_1 = st.button(label='Submit Prompt')
    if submit_button_1:

        ### OpenAI API code - obain a key and account here.###
        os.environ["OPENAI_API_KEY"] = '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'

        ### Begin GPT-3 text embedding search. Code and explanation for GPT-3 embeddgins found here: ###
        datafile_path = "./more_index_embeddings.csv"  # for your convenience, we precomputed the embeddings
        df = pd.read_csv(datafile_path)
        df["babbage_search"] = df.babbage_search.apply(eval).apply(np.array)

        def search_text(df, product_description, n=3, pprint=True):
            embedding = get_embedding(
                product_description,
                engine="text-search-babbage-query-001"
            )
            df["similarities"] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

            res = (
                df.sort_values("similarities", ascending=False)
                .head(n)
                .combined.str.replace("Summary: ", "")
                .str.replace("Text:", ": ")
            )
            if pprint:
                for r in res:
                    print(r[:500])
                    print()
            return res

        res = search_text(df, submission_text, n=3)

        ### Begin GPT-3 prompting ###
        ### Prompt 1: Text Identifier
        id_prompt = './more-identified-COT_formatted.txt'

        def get_text(id_prompt):
            with open(id_prompt, "r", encoding='utf8') as file:
                text = file.read()
                return(text)

        txt_identifer = get_text(id_prompt)

        openai.api_key = os.getenv("OPENAI_API_KEY")

        summon = openai.Completion.create(
                model='text-davinci-002',
                prompt= txt_identifer + "/nExcellent. Let's try another./nQuestion:" + submission_text + "/nSections:/n " + res + "/nIdentification: ",
                temperature=temperature_dial,
                max_tokens=response_length)

        response_json = len(summon["choices"])

        for item in range(response_json):
            output = summon['choices'][item]['text']

        output_cleaned = output.replace("\n", "")
        output_cleaned2 = output_cleaned.strip()

        d = {'prompt':[submission_text], 'output':[output_cleaned2], 'temperature':[temperature_dial*10], 'response_length':[response_length]}
        df = pd.DataFrame(data=d, index=None)
        df.to_csv('./more_extractor_outputs.csv',mode='a', index=True)

        st.write(output2)
        st.write("/n/n")
        #st.subheader('Please rank this reply for future improvement.')
        st.write("Here are the most relevant sections of the text I identified:")
        st.write("/n")
        st.write(res)
