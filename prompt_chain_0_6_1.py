from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
import openai
import csv
from datetime import datetime as dt
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
import re
import streamlit as st
import pygsheets
from google.oauth2 import service_account
import ssl
import base64


scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)


st.set_page_config(
    page_title="Ask a Source: More's History of Richard III",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")



st.title("Ask A Source: Thomas More's 'The History of Richard III'")
col1, col2 = st.columns([3.0,3])
with col1:
    #book_pic = st.image(image ='./more_page.jpg', caption="From Thomas More's 'History of Richard III' (1557). British Library.", width=500)
    #st.write("Explore the current data.")
    #df = pd.read_csv('richardbot1_data.csv')
    #st.dataframe(df, height=500)
    #st.markdown("""
    #<embed src="https://thomasmorestudies.org/wp-content/uploads/2020/09/Richard.pdf" width="800" height="800">
    #""", unsafe_allow_html=True)
    #pdf_display = F'<iframe src="https://thomasmorestudies.org/wp-content/uploads/2020/09/Richard.pdf" width="700" height="1000" type="application/pdf"></iframe>'
    pdf_url = 'https://github.com/Dr-Hutchinson/prompt_chain_0/blob/main/annotated_full_text.pdf'

    pdf_display = F'<iframe src="{pdf_url}" width="700" height="1000" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)



        #st_display_pdf("C:\\Users\\danie\\Desktop\\AI_Art\\GPT-2\\history of richard iii\\Streamlit\\prompt_chain_0\\prompt_chain_0\\annotated_full_text.pdf")

def button_one():
    st.write("This application uses GPT-3 to answer questions about Thomas More's [_History of King Richard III_](https://thomasmorestudies.org/wp-content/uploads/2020/09/Richard.pdf). Choose one of the options below, and pose a question about the text.")
    semantic_search = "Semantic Search: Enter a question, and recieve sections of the text that are the most closely related."
    ask_a_paragraph = "Ask a Paragraph: Select a Section from the text, and then pose a question. GPT-3 will search the internet answer your question."
    ask_a_source = "Ask A Source: Pose a question about the text, and GPT-3 will share answers drawn from the text along with historical analysis."


    search_method = st.radio("Choose a method:", (semantic_search, ask_a_paragraph, ask_a_source))
    section_number = st.number_input('Select a section number if you have selected Ask a Paragraph. You can find the section numbers either through semantic search, or via this link.')
    submission_text = st.text_area("Enter your question below. ")
    submit_button_1 = st.button(label='Click here to submit your question.')
    if submit_button_1:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

        def embeddings_search():
            datafile_path = "./more_index_embeddings.csv"
            df = pd.read_csv(datafile_path)
            df["babbage_search"] = df.babbage_search.apply(eval).apply(np.array)

            def search_text(df, product_description, n=3, pprint=True):
                embedding = get_embedding(
                    product_description,
                    engine="text-search-babbage-query-001"
                )
                df["similarities"] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

                    # Select the first three rows of the sorted DataFrame
                top_three = df.sort_values("similarities", ascending=False).head(3)

                    # If `pprint` is True, print the output
                if pprint:
                    for _, row in top_three.iterrows():
                        print(row["combined"][:500])
                        print()

                    # Return the DataFrame with the added similarity values
                return top_three

                # Call the search_text() function and store the return value in a variable
            results_df = search_text(df, submission_text, n=3)

                # Reset the index and create a new column "index"
            results_df = results_df.reset_index()

                # Access the values in the "similarities" and "combined" columns
            similarity1 = results_df.iloc[0]["similarities"]
            combined1 = str(results_df.iloc[0]["combined"])

            similarity2 = results_df.iloc[1]["similarities"]
            combined2 = results_df.iloc[1]["combined"]

            similarity3 = results_df.iloc[2]["similarities"]
            combined3 = results_df.iloc[2]["combined"]

            # Write the DataFrame to a CSV file
            results_df.to_csv('results_df.csv', index=False, columns=["similarities", "combined"])

            num_rows = results_df.shape[0]

            # Iterate through the rows of the dataframe
            for i in range(num_rows):
              # Get the current row
              row = results_df.iloc[i]

              # Create an expander for the current row, with the label set to the row number
              with st.expander(label="Text Section  " + str(i) + ":", expanded=True):
                # Display each cell in the row as a separate block of text
                st.markdown("**Question:**")
                st.write(submission_text)
                st.markdown("**Below is a section of the text along with its semantic similarity score. It is one of the three highest scoring sections in the text. Semantic similaritiy scores above .30 are generally relevant.**")
                st.write(row['similarities'])
                st.write(row['combined'])

                #st.write(row['initial_analysis'])
                #st.markdown("Biographical Identification: \n\n" + row['final_analysis'])


        def self_ask_with_search():

            datafile_path = "./more_index_combined.csv"

            df = pd.read_csv(datafile_path, encoding='latin1')
            section_select = r"Summary: Section_{}(:|$)".format(section_number)
            result = df[df['combined'].str.contains(section_select, regex=True)]
            # Select the 'combined' column of the result DataFrame
            result = result.loc[:, 'combined']
            # Do something with the resulting cell value
            section = result.iloc[0]


            # self-search experiment 0.3 - revised question prompt examples
            dertford_question = "2. Text:\nSummary: Section_0: King Edward IV died in 1483, leaving behind seven children. Edward, the eldest, was 13 years old at the time of his father's death. Richard, the second son, was two years younger. Elizabeth, Cecily, Brigette, Anne, and Katherine were the King's daughters. Elizabeth was later married to King Henry VII, and Anne was married to Thomas Howard, Earl of Surrey. Katherine was the last of the King's children to marry, and she eventually married a man of wealth. Text: Section_0: King Edward of that name the Fourth, after he had lived fifty and three years, seven months, and six days, and thereof reigned two and twenty years, one month, and eight days, died at Westminster the ninth day of April, the year of our redemption, a thousand four hundred four score and three, leaving much fair issue, that is, Edward the Prince, thirteen years of age; Richard Duke of York, two years younger; Elizabeth, whose fortune and grace was after to be queen, wife unto King Henry the Seventh, and mother unto the Eighth; Cecily not so fortunate as fair; Brigette, who, representing the virtue of her whose name she bore, professed and observed a religious life in Dertford, a house of cloistered Nuns; Anne, who was after honorably married unto Thomas, then Lord Howard and after Earl of Surrey; and Katherine, who long time tossed in either fortune, sometime in wealth, often in adversity, at the last, if this be the last, for yet she lives, is by the goodness of her nephew, King Henry the Eighth, in very prosperous state, and worthy her birth and virtue.\n3. Object of the Question: The religious order associated with Dertford\n4. Historical Context: 15th century England\n5. Revised User Question: What was the religious order associated with Dertford in 15th century England?\nExcellent, let's try another."
            ludlow_question = "2. Text:\nSummary: Section_17: After King Edward IV's death, his son Prince Edward moved towards London. He was accompanied by Sir Anthony Woodville, Lord Rivers, and other members of the queen's family. Text: Section_17: As soon as the King was departed, that noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales. Such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected. And for this reason the Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages. To the governance and ordering of this young Prince, at his sending thither, was there appointed Sir Anthony Woodville, Lord Rivers and brother unto the Queen, a right honorable man, as valiant of hand as politic in counsel. Adjoined were there unto him others of the same party, and, in effect, every one as he was nearest of kin unto the Queen was so planted next about the Prince.\n3. Object of the Question: The status of Ludlow as a castle\n4. Historical Context: 15th century England\n5. Revised User Question: Was Ludlow a castle in 15th century England, and if so, what was its purpose?\nExcellent, let's try another."
            malmsey_question = "2. Text\nSummary: Section_7: George, Duke of Clarence, was accused of treason and sentenced to death. He was drowned in a butt of malmesey, and his death was piteously bewailed by King Edward IV. Text: Section_7: George, Duke of Clarence, was a goodly noble prince, and at all points fortunate, if either his own ambition had not set him against his brother, or the envy of his enemies had not set his brother against him. For were it by the Queen and the lords of her blood, who highly maligned the King's kindred (as women commonly, not of malice but of nature, hate them whom their husbands love), or were it a proud appetite of the Duke himself intending to be king, in any case, heinous treason was there laid to his charge, and, finally, were he faulty or were he faultless, attainted was he by Parliament and judged to the death, and thereupon hastily drowned in a butt of malmesey, whose death, King Edward (although he commanded it), when he knew it was done, piteously bewailed and sorrowfully repented.\n3. Object of the Question: The nature and use of a butt of malmsey\n4. Historical Context: 15th century England\n5. Revised User Question: In 15th century England, what was a butt of malmsey and how was it used?\nExcellent. Let's try another."
            seal_question = "2. Text\nSummary: Section_33: At a council meeting, Duke of Gloucester is made Protector and Archbishop of York is reproved. The Lord Chamberlain and some others keep their offices. Text: Section_33: But the Duke of Gloucester bore himself in open sight so reverently to the Prince, with all semblance of lowliness, that from the great obloquy in which he was so late before, he was suddenly fallen in so great trust, that at the Council next assembled, he was the only man chosen and thought most suitable to be Protector of the King and his realm, so that were it destiny or were it folly the lamb was given to the wolf to keep. At which Council also the Archbishop of York, Chancellor of England, who had delivered up the Great Seal to the Queen, was thereof greatly reproved, and the Seal taken from him and delivered to Doctor Russell, Bishop of Lincoln, a wise man and good and of much experience, and one of the best learned men undoubtedly that England had in his time. Diverse lords and knights were appointed unto diverse offices. The Lord Chamberlain and some others kept still their offices that they had before.\n3. Object of the Question: The purpose and function of the Great Seal\n4. Specify the Historical Context of the Text and User Question: The text is describing events that took place in the late 15th century in England.\n5. Compose a Revised Question: What was the Great Seal in late 15th century England?\nExcellent, let's try another."
            anjou_question = "2. Text:\nSummary: Section_6: Richard, Duke of York, a noble man and a mighty, had begun not by war but by law to challenge the crown, putting his claim into the Parliament. There his cause was either for right or favor so far forth advanced that King Henry (although he had a goodly prince [Edward, son by Margaret of Anjou]) utterly rejected his own blood ; the crown was by authority of Parliament entailed unto the Duke of York, and his male issue in remainder, immediately after the death of King Henry. But the Duke, not enduring so long to tarry, but intending under pretext of dissension and debate arising in the realm, to reign before his time and to take upon him the rule in King Henry's life, was with many nobles of the realm at Wakefield slain, leaving three sons Edward, George, and Richard.\n3. Object of the Question: Margaret of Anjou's identity and role or position\n4. Historical Context: 15th century England\n5. Revised User Question: Who was Margaret of Anjou in 15th century England?\nExcellent. Let's try another."

            # # self-search experiment 0.3 - revised question prompt template
            examples = [
                {"question": "1. User Question: What religious order was associated with Dertford?", "output": dertford_question},
                {"question": "1. User Question: Was Ludlow a castle?", "output": ludlow_question},
                {"question": "1. User Question: What is a butt of malmesey?", "output": malmsey_question},
                {"question": "1. User Question: What is the Great Seal?", "output": seal_question},
                {"question": "1. User Question: Who was Margaret of Anjou?", "output": anjou_question},
            ],
            # This how we specify how the example should be formatted.
            example_prompt = PromptTemplate(
                input_variables=["question"],
                template="question: {question}",
            )

            # # self-search experiment 0.3 - revised question prompt template
            #don't delete

            prompt_from_string_examples5 = FewShotPromptTemplate(
                # These are the examples we want to insert into the prompt.
                examples=examples,
                # This is how we want to format the examples when we insert them into the prompt.
                example_prompt=example_prompt,
                # The prefix is some text that goes before the examples in the prompt.
                # Usually, this consists of intructions.
                prefix="You are an AI with expertise in Thomas More's 'History of Richard III' and the larger political, cultural, and social history of England during the War of the Roses. In this exercise you will be given a selected section of More's text (known as the Text), a User Question, and a Method for historically contextualizing and rephrasing the User Question.  The goal of this exercise is to use this information to revise the User Question for historical contextualization.\nHere is your Method. Let's take it step by step.\n1. User Question: You will first be given a question about the Text by the user.\n2. Text: You will then be given a section of the Text selected by a user.\n3. Object of the Question: Specify what the user is seeking to learn in the User Question.\n4. Specify the Historical Context of the Text and User Question: Consider the specific historical context of the Text and the User Question, such as the time period (e.g. late 15th century) or geographic location (England).\n5. Compose a Revised User Question: Adjust the User Question to reflect the historical context noted in the previous step, but maintain the Object of the Question as the main focus. Ideally the usual format of much revisions would be 'User Question' + 'Historical Context'.  For example, the question 'What was the capital of England' would be revised to 'What was the capital of England in 15th century England'. Adapt the User Question in a similar manner.\nLet's begin.",
                # The suffix is some text that goes after the examples in the prompt.
                # Usually, this is where the user input will go
                suffix="Question: {question}\n3. Object of the Question\n",
                # The input variables are the variables that the overall prompt expects.
                input_variables=["question"],
                # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                example_separator="\n\n"
            )

            llm = OpenAI(model_name="text-davinci-003", max_tokens = 500, temperature=0.0)
            chain1 = LLMChain(llm=llm, prompt=prompt_from_string_examples5)
            knowledge_check = chain1.run(submission_text+section)

            # pass on revised quesiton for search

            lines = knowledge_check.split('\n')

            # Search for the line that starts with "Compose a Revised User Question"
            for i, line in enumerate(lines):
              if line.startswith("5. Compose a Revised User Question"):
                revised_question = lines[i+1]

            llm = OpenAI(temperature=0)
            search = SerpAPIWrapper()
            tools = [
                Tool(
                    name="Intermediate Answer",
                    func=search.run
                )
            ]

            self_ask_with_search = initialize_agent(tools, llm, agent="self-ask-with-search", verbose=True, return_intermediate_steps=True)
            init_reasoning = self_ask_with_search({"input": revised_question})

            reasoning = ""
            for i, step in enumerate(init_reasoning['intermediate_steps']):
              action = step[0]
              output = step[1]
              reasoning += f"{action.tool_input}: {output}\n"

            # final report prompt example

            dertford_question = "2. Text:\nSummary: Section_0: King Edward IV died in 1483, leaving behind seven children. Edward, the eldest, was 13 years old at the time of his father's death. Richard, the second son, was two years younger. Elizabeth, Cecily, Brigette, Anne, and Katherine were the King's daughters. Elizabeth was later married to King Henry VII, and Anne was married to Thomas Howard, Earl of Surrey. Katherine was the last of the King's children to marry, and she eventually married a man of wealth. Text: Section_0: King Edward of that name the Fourth, after he had lived fifty and three years, seven months, and six days, and thereof reigned two and twenty years, one month, and eight days, died at Westminster the ninth day of April, the year of our redemption, a thousand four hundred four score and three, leaving much fair issue, that is, Edward the Prince, thirteen years of age; Richard Duke of York, two years younger; Elizabeth, whose fortune and grace was after to be queen, wife unto King Henry the Seventh, and mother unto the Eighth; Cecily not so fortunate as fair; Brigette, who, representing the virtue of her whose name she bore, professed and observed a religious life in Dertford, a house of cloistered Nuns; Anne, who was after honorably married unto Thomas, then Lord Howard and after Earl of Surrey; and Katherine, who long time tossed in either fortune, sometime in wealth, often in adversity, at the last, if this be the last, for yet she lives, is by the goodness of her nephew, King Henry the Eighth, in very prosperous state, and worthy her birth and virtue.\n3.Object of the Question: The religious order associated with Dertford\n4. Historical Context: 15th century England\n5. Revised User Question: What was the religious order associated with Dertford in 15th century England?\n6. Research Summary: \nWhat is Dertford?: Dartford is the principal town in the Borough of Dartford, Kent, England. It is located 18 miles south-east of Central London and is situated adjacent to the London Borough of Bexley to its west. To its north, across the Thames estuary, is Thurrock in Essex, which can be reached via the Dartford Crossing.\nWhat religious order was associated with Dartford in the 15th century?: The priory of Dartford was the only house of Dominican nuns, or ' Sisters of the Order of St. Augustine according to the institutes and under the care of ...\n7. Research Summary Answer: The Order of St. Augustine\n8.Final Report: Answer: In 15th century England, the religious order associated with Dertford was the Dominicans, also known as the Order of Preachers.\nConfidence Level: Medium\nDetails: In Thomas More's 'History of Richard III', it is mentioned that Brigette, one of King Edward IV's daughters, 'professed and observed a religious life in Dertford, a house of cloistered Nuns.' This information is supported by research which indicates that Dertford was a town located in Kent, England and that it was home to a number of religious institutions, including nunneries. However, it is also possible that the 'house of cloistered Nuns' referred to in the text was actually a convent of Dominican friars, as research indicates that Dertford had a Dominican convent during this time period. The Dominicans were a Catholic religious order founded by Saint Dominic in the early 13th century, and they were known for their commitment to preaching and teaching the Gospel. It is possible that the Dominicans, who were not traditionally associated with cloistered communities, adopted the term 'cloistered Nuns' to describe their convent in Dertford. It is important to note that the text does not specify the religious order to which Brigette belonged, so this assumption is based on the information available about the Dominicans in Dertford and the possibility that the reference to 'cloistered Nuns' may not be literal.\nExcellent, let's try another."
            ludlow_question = "2. Text\nSummary: Section_17: After King Edward IV's death, his son Prince Edward moved towards London. He was accompanied by Sir Anthony Woodville, Lord Rivers, and other members of the queen's family. Text: Section_17: As soon as the King was departed, that noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales. Such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected. And for this reason the Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages. To the governance and ordering of this young Prince, at his sending thither, was there appointed Sir Anthony Woodville, Lord Rivers and brother unto the Queen, a right honorable man, as valiant of hand as politic in counsel. Adjoined were there unto him others of the same party, and, in effect, every one as he was nearest of kin unto the Queen was so planted next about the Prince.\n3. Knowledge Report:\nPreliminary Answer: The Text does not mention whether Ludlow was a castle.\nConsider Details Needed to Answer the Question: Information about the physical characteristics and location of Ludlow.\nDetermine Your Knowledge of these Details: Low confidence.\nDetail Assessment: Insufficient detail for Answer.\nCompose a question for further research: Was Ludlow a castle in the late 15th century?\n4. Research Summary:\nWhere is Ludlow located?: Ludlow is a market town in Shropshire, England. The town is significant in the history of the Welsh Marches and in relation to Wales. It is located 28 miles …\nWas there a castle in Ludlow?: Throughout the 16th and 17th centuries, Ludlow Castle was held by the Crown, except for a brief time during the Civil War and the Commonwealth. It enjoyed great status as the centre of administration for the Marches shires and for Wales – court sessions and the Prince's Council were held here.\n5. Research Summary Answer: Yes, Ludlow was a castle in the late 15th century.\n6. Final Report with all three elements:\nAnswer: Ludlow was a castle in the late 15th century.\nConfidence Level: High\nDetails: Ludlow Castle was located in the market town of Shropshire, England. It was held by the Crown throughout the 16th and 17th centuries, except for a brief period during the Civil War and the Commonwealth. Ludlow Castle was known as the centre of administration for the Marches shires and Wales, and court sessions and the Prince's Council were held there. According to the Text, 'such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected. And for this reason the Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages.' (S.17) This suggests that Ludlow Castle was seen as a place of authority and order in a region that was prone to lawlessness.\nExcellent, let's try another."
            seal_question = "2. Text\nSummary: Section_33: At a council meeting, Duke of Gloucester is made Protector and Archbishop of York is reproved. The Lord Chamberlain and some others keep their offices. Text: Section_33: But the Duke of Gloucester bore himself in open sight so reverently to the Prince, with all semblance of lowliness, that from the great obloquy in which he was so late before, he was suddenly fallen in so great trust, that at the Council next assembled, he was the only man chosen and thought most suitable to be Protector of the King and his realm, so that were it destiny or were it folly the lamb was given to the wolf to keep. At which Council also the Archbishop of York, Chancellor of England, who had delivered up the Great Seal to the Queen, was thereof greatly reproved, and the Seal taken from him and delivered to Doctor Russell, Bishop of Lincoln, a wise man and good and of much experience, and one of the best learned men undoubtedly that England had in his time. Diverse lords and knights were appointed unto diverse offices. The Lord Chamberlain and some others kept still their offices that they had before.\n3. Object of the Question: What was the function and purpose of the Great Seal.\n4. Historical Context: The text is describing events that took place in the late 15th century in England.\n5. Revised Question: What was the Great Seal in late 15th century England?\n 6. Research Summary: \nWhat is the Great Seal?: The Great Seal is a principal national symbol of the United States. The phrase is used both for the physical seal itself, which is kept by the United States Secretary of State, and more generally for the design impressed upon it.\nWhat was the purpose of the Great Seal in 15th century England?: The seal meant that the monarch did not need to sign every official document in person; authorisation could be carried out instead by an appointed officer. In centuries when few people could read or write, the seal provided a pictorial expression of Royal approval which all could understand.\n7. Research Summary Answer: The Great Seal was used to authorize documents on behalf of the monarch in 15th century England.\n8. Final Report:\nAnswer: In late 15th century England, the Great Seal was used to authorize documents on behalf of the monarch.\nConfidence Level: High\nDetails: The research summary states that the Great Seal was a symbol of national importance used to authorize documents in the absence of the monarch's personal signature. This information is supported by the text, which mentions that 'the Archbishop of York, Chancellor of England, who had delivered up the Great Seal to the Queen, was thereof greatly reproved, and the Seal taken from him and delivered to Doctor Russell, Bishop of Lincoln, a wise man and good and of much experience.' (S.33) This suggests that the Great Seal held significant power and authority and was entrusted to those who were responsible for carrying out the monarch's wishes. The transfer of the Great Seal from the Archbishop of York to Doctor Russell further emphasizes the importance of the seal and its role in the authorization of official documents.\nExcellent. Let's try another."
            anjou_question = "2. Text:\nSummary: Section_6: Richard, Duke of York, was killed in battle at Wakefield, leaving behind three sons: Edward, George, and Richard. Edward IV then usurped the crown. Text: Section_6: Richard, Duke of York, a noble man and a mighty, had begun not by war but by law to challenge the crown, putting his claim into the Parliament. There his cause was either for right or favor so far forth advanced that King Henry (although he had a goodly prince [Edward, son by Margaret of Anjou]) utterly rejected his own blood ; the crown was by authority of Parliament entailed unto the Duke of York, and his male issue in remainder, immediately after the death of King Henry. But the Duke, not enduring so long to tarry, but intending under pretext of dissension and debate arising in the realm, to reign before his time and to take upon him the rule in King Henry's life, was with many nobles of the realm at Wakefield slain, leaving three sons Edward, George, and Richard.\n3. Knowledge Report:\nPreliminary Answer: Margaret of Anjou was the queen consort of King Henry VI of England.\nConsider Details Needed to Answer the Question: None.\nDetermine Your Knowledge of these Details: High.\nDetail Assessment: Sufficient detail for Answer.\nAnswer: Margaret of Anjou was the queen consort of King Henry VI of England.\nCompose a question for further research: What role did Margaret of Anjou play in the Wars of the Roses and the succession of the English throne?\n4. Research Summary:\nWho was Margaret of Anjou?: Margaret of Anjou was Queen of England and nominally Queen of France by marriage to King Henry VI from 1445 to 1461 and again from 1470 to 1471. Born in the Duchy of Lorraine into the House of Valois-Anjou, Margaret was the second eldest daughter of René, King of Naples, and Isabella, Duchess of Lorraine.\nWhat role did Margaret of Anjou play in the Wars of the Roses?: Margaret of Anjou was one of the major players in the Wars of the Roses. She often led the Lancastrian forces during the wars and dictated grand strategy. She battled her arch enemy Richard, duke of York over the royal succession and unsuccessful tried to place her son, Edward, on the throne.\nWhat role did Margaret of Anjou play in the succession of the English throne?: Margaret took the lead in attempting to restore the house of Lancaster to the throne. From Scotland, she sought assistance from nobles there and sent envoys to France. In England there remained some nobles who were loyal to Henry and the Lancastrian cause.\n5. Research Summary Answer: Margaret of Anjou was one of the major players in the Wars of the Roses and attempted to restore the house of Lancaster to the throne by seeking assistance from nobles in Scotland and France and rallying those who remained loyal to Henry and the Lancastrian cause.\n6. Final Report with all three elements:\nAnswer: Margaret of Anjou was Queen of England and nominally Queen of France by marriage to King Henry VI. She played a major role in the Wars of the Roses and attempted to restore the house of Lancaster to the throne.\nConfidence: High\nDetails: More's Text states that Richard, Duke of York, ‘began not by war but by law to challenge the crown, putting his claim into the Parliament.’ (S. 6) In Parliament, his cause ‘was either for right or favor so far forth advanced that King Henry (although he had a goodly prince [Edward, son by Margaret of Anjou]) utterly rejected his own blood.’ (S.6) This demonstrates the importance of Margaret and her son in the succession struggle and conflict with the house of York. The Text also mentions that the crown was ‘by authority of Parliament entailed unto the Duke of York, and his male issue in remainder, immediately after the death of King Henry.’ (S.6) Margaret attempted to restore the house of Lancaster to the throne and played a prominent role in the Wars of the Roses, leading the Lancastrian forces and dictating grand strategy. She fought against her arch enemy Richard, Duke of York, in an effort to place her son Edward on the throne. In her quest to reclaim the throne, Margaret sought support from nobles in Scotland and sent envoys to France. There were still some nobles in England who remained loyal to Henry and the Lancastrian cause."

            # final answer prompt template
            examples = [
                {"question": "1. Question: What religious order was associated with Dertford?", "output": dertford_question},
                {"question": "1. Question: Who was Margaret of Anjou?\n", "output": anjou_question},
                {"question": "1. Question: Was Ludlow a castle?\n", "output": ludlow_question},
                {"question": "1. Question: Was was the Great Seal?\n", "output": seal_question},
            ],
            # This how we specify how the example should be formatted.
            example_prompt = PromptTemplate(
                input_variables=["question"],
                template="question: {question}",
            )

            # final answer prompt template
            #don't delete

            prompt_from_string_examples6 = FewShotPromptTemplate(
                # These are the examples we want to insert into the prompt.
                examples=examples,
                # This is how we want to format the examples when we insert them into the prompt.
                example_prompt=example_prompt,
                # The prefix is some text that goes before the examples in the prompt.
                # Usually, this consists of intructions.
                prefix="You are an AI with expertise in Thomas More's 'History of Richard III' and the larger political, cultural, and social history of England during the War of the Roses. In this exercise you will be given a researcher's process for obtaining information to answer questions about More's Text. Your job is to compose a final report based on the information obtained by the researcher. You will compose this report based on the Method below.\nHere is your Method. Let's take it step by step.\n1. User Question: A question about the Text by a user.\n2. Text: A section of the Text selected by a user.\n3. Object of the Question: What the user is seeking to learn in the User Question.\n4. Historical Context of the Text and User Question:  The specific historical context of the Text and the User Question, such as the time period (e.g. late 15th century) or geographic location (England).\n5. Revised User Question: A revised version of the User Question adapted to the historical context identified above.\n6. Research Summary: A summary of outside research conducted by the researcher to provide more detail to answer the Revised User Question. This summary contains sub-questions composed by the researcher to answer the Revised User Question, and a summary of information the researcher uncovered in exploring these sub-questions.\n7. Research Summary Answer: The researcher's proposed answer to the Revised User Question based on the information contained in the Research Summary.\n8. Final Report: You mission is to compose a Final Report based on information in the Research Summary and the Text. This report is distinct and different from the Revised User Question. The Final Report includes three elements: an Answer to the User Question, confidence level in the answer (high, medium, low), and a Details section outlining the evidence and connecting it with a supporting quote from the Text. To be complete the Final Report must contain all three of these elements, and it is important to complete the Final Report.",
                # The suffix is some text that goes after the examples in the prompt.
                # Usually, this is where the user input will go
                suffix="Question: {question}\8. Final Report\n",
                # The input variables are the variables that the overall prompt expects.
                input_variables=["question"],
                # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                example_separator="\n\n"
            )

            llm = OpenAI(model_name="text-davinci-003", max_tokens = 1000, temperature=0.0)
            chain2 = LLMChain(llm=llm, prompt=prompt_from_string_examples6)
            final_answer = chain2.run(submission_text + "\n" + section + "\n3. Object of the Question\n" + knowledge_check + "\n6. Research Summary\n" + reasoning + "\n7. Research Summary Answer:\n" + output)
            st.write()

        def ask_a_source():

            st.header("GPT-3's analysis is underway. It can take a minute or two for every step of the process to be completed. GPT-3's progress will be documented below.")

                            ### embeddings search
                #begin code

                #cell for running OpenAI embeddings on csv file.
                #base embeddings search w/ similarity values - version_1 - works
            #begin code
                #cell for running OpenAI embeddings on csv file.

                #datafile_path = "./more_index_embeddings.csv"  # for your convenience, we precomputed the embeddings
            datafile_path = "./more_index_embeddings.csv"
            df = pd.read_csv(datafile_path)
            df["babbage_search"] = df.babbage_search.apply(eval).apply(np.array)

            def search_text(df, product_description, n=3, pprint=True):
                embedding = get_embedding(
                    product_description,
                    engine="text-search-babbage-query-001"
                )
                df["similarities"] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))

                    # Select the first three rows of the sorted DataFrame
                top_three = df.sort_values("similarities", ascending=False).head(3)

                    # If `pprint` is True, print the output
                if pprint:
                    for _, row in top_three.iterrows():
                        print(row["combined"][:500])
                        print()

                    # Return the DataFrame with the added similarity values
                return top_three

                # Call the search_text() function and store the return value in a variable
            results_df = search_text(df, submission_text, n=3)

                # Reset the index and create a new column "index"
            results_df = results_df.reset_index()

                # Access the values in the "similarities" and "combined" columns
            similarity1 = results_df.iloc[0]["similarities"]
            combined1 = str(results_df.iloc[0]["combined"])

            similarity2 = results_df.iloc[1]["similarities"]
            combined2 = results_df.iloc[1]["combined"]

            similarity3 = results_df.iloc[2]["similarities"]
            combined3 = results_df.iloc[2]["combined"]

            st.write("Step 1 complete - identified the most semantically similar text sections.")
                # Write the DataFrame to a CSV file
            #results_df.to_csv('results_df.csv', index=False, columns=["similarities", "combined"])
                #end code

                ### few_shot examples for text relevance prompt
                #relevance_check_w_similairities.0 (prompts w/ similarities & probability)

            burial_question = "2. Section: Summary: Section_160:  Sir James had the murderers bury King Edward V and Prince Richard's bodies deep in the ground under a heap of stones. Text: Section_160: Which after that the wretches perceived, first by the struggling with the pains of death, and after long lying still, to be thoroughly dead, they laid their bodies naked out upon the bed, and fetched Sir James to see them. Who, upon the sight of them, caused those murderers to bury them at the stair-foot, suitably deep in the ground, under a great heap of stones..\n3. SSS: 0.396\n4.Key Words: Edward V, Prince Richard, bodies, bury, Sir James\n5. Background knowledge and context: Edward V was one of the sons of King Edward IV and Prince Richard was his brother. Sir James was involved in their deaths and had their bodies buried.\n6.Relevance Determination: Medium\n7. Relevance Explanation: The key words 'Edward V' and 'Prince Richard' are related to the question as they are mentioned in the same sentence as 'bury'. However, the question specifically asks about the burial of Edward IV, not Edward V and Prince Richard.\n8Final Output: Section_160: Irrelevant.\nExcellent. Let's try another.",
            cecily_question = "2. Section: Summary: Section_113:  In a sermon at Paul's Cross, it was revealed to the people that King Edward IV's marriage was not lawful, and that his children were bastards. Text: Section_113: Now then as I began to show you, it was by the Protector and his council concluded that this Doctor Shaa should in a sermon at Paul's Cross signify to the people that neither King Edward himself nor the Duke of Clarence were lawfully begotten, nor were the very children of the Duke of York, but gotten unlawfully by other persons by the adultery of the Duchess, their mother, and that also Dame Elizabeth Lucy was verily the wife of King Edward, and so the Prince and all his children were bastards that were gotten upon the Queen.\n3. SSS: 0.369\nBased on the provided information, it appears that the section is potentially relevant to the question. The semantic similarity score is relatively high, indicating that there may be some connection between the section and the question. However, it is important to carefully examine the section and the question to determine the specific relevance.\n4.Key Words: The key words in the section that may be specifically and directly related to the question are 'King Edward,' 'Duke of Clarence,' 'Duke of York,' 'Elizabeth Lucy,' 'Prince,' and 'children.' These words refer to individuals or groups of people mentioned in the section.\n5. Background knowledge and context: Knowing that the question is asking about a person named Cecily, we can use our background knowledge about the context of the text to further assess the relevance of the section. The section mentions several individuals and groups of people, including King Edward, the Duke of Clarence, the Duke of York, Elizabeth Lucy, the Prince, and the children. Cecily is not mentioned by name in the section.\n6.Relevance Determination: Based on the key words identified in the section and our background knowledge of the context, it is unlikely that the section is relevant to the question. The section does not mention the name Cecily and does not provide any information about her. Therefore, I have a low degree of confidence in determining that the section is relevant to the question.\n7.Relevance Explanation: The section is not relevant to the question because it does not mention the name Cecily and does not provide any information about her.\n8.Final Output: Section_113: Irrelevant.\nExcellent. Let's try another.",
            edward_question = "2. Section:Summary: Section_3:  King Edward IV was a good-looking and strong man who was wise in counsel and just in war. He was also known for his love of women and good food. However, he was also known to be a fair and merciful man, and he was greatly loved by his people. Text: Section_3: He was a goodly personage, and very princely to behold: of heart, courageous; politic in counsel; in adversity nothing abashed; in prosperity, rather joyful than proud; in peace, just and merciful; in war, sharp and fierce; in the field, bold and hardy, and nevertheless, no further than wisdom would, adventurous. Whose wars whosoever would well consider, he shall no less commend his wisdom when he withdrew than his manhood when he vanquished. He was of visage lovely, of body mighty, strong, and clean made; however, in his latter days with over-liberal diet , he became somewhat corpulent and burly, and nonetheless not uncomely; he was of youth greatly given to fleshly wantonness, from which health of body in great prosperity and fortune, without a special grace, hardly refrains. This fault not greatly grieved the people, for one man's pleasure could not stretch and extend to the displeasure of very many, and the fault was without violence, and besides that, in his latter days, it lessened and well left.\n3. SSS: 0.428\nTo determine whether this section is relevant to the question, let's follow the steps of the Method:1.Question: The user's question is ‘What was King Edward IV's appearance?’\n2.Section: The given section is about King Edward IV's appearance, character, and behavior.\n3. SSS: The semantic similarity score (SSS) is 0.428, which is above the threshold of .40 and indicates that there is some potential relevance between the section and the question.\n4. Key Words: Key words in the section that are directly and specifically related to the question include ‘goodly personage,’ ‘visage lovely,’ ‘body mighty, strong, and clean made,’ and ‘somewhat corpulent and burly.’ These words directly describe King Edward IV's appearance.\n5. Background Knowledge: Based on my background knowledge of the subject matter, I can confirm that this section is directly and specifically relevant to answering the question about King Edward IV's appearance.\n6. Relevance Determination: The relevance determination is high, as the section is directly and specifically related to the question.\n7. Relevance Explanation: The relevance explanation is that the section contains detailed descriptions of King Edward IV's appearance, including his physical appearance and any changes to it over time.\n8. Final Output: Therefore, the final output is ‘Section_3: Relevant.’\nExcellent. Let's try another."

                #### langchain few-shot prompting
                #relevance_check_w_similairities.0 (prompts w/ similarities & probability)

                # These are some examples of a pretend task of creating antonyms.
            examples = [
                {"question": "1. Question: Where was Edward IV buried?", "output": burial_question},
                {"question": "1. Question: What was Edward IV's appearence?", "output": edward_question},
                {"question": "1. Question: Who is Cecily?", "output": cecily_question}
            ],
                # This how we specify how the example should be formatted.
            example_prompt = PromptTemplate(
                input_variables=["question"],
                template="question: {question}",
            )
                ### langchain prompt loading
                #relevance_check_w_similairities.0 (prompts w/ similarities & probability)

            prompt_from_string_examples = FewShotPromptTemplate(
                # These are the examples we want to insert into the prompt.
                examples=examples,
                # This is how we want to format the examples when we insert them into the prompt.
                example_prompt=example_prompt,
                # The prefix is some text that goes before the examples in the prompt.
                # Usually, this consists of intructions.
                prefix="You are an AI expert on the 'History of Richard III' by Thomas More. In this exercise you are given a user supplied question, a Section of the Text, a Semantic Similarity Score, and a Method for determining the Section’s relevance to the Question. Your objective is to determine whether that Section of the text is directly and specifically relevant to the user question. You will be the Method below to fulfill this objective, taking each step by step.\n\nHere is your Method.\nMethod: Go step by step in answering the question.\n1. Question: You will be provided with a user question.\n2. Section: You will be given a section of the text from Thomas More's 'The History of Richard III.' \n3. Semantic Similarity Score: You are then given a semantic similarity score, which ranges from 1.0 (highest) to 0.0 (lowest). The higher the score, the more likely its potential relevance. Scores approaching .40 and above are generally considered to have some relevance. However, this score isn’t fully determinative, as other semantically related words in the Section can generate false positives. Weigh the value of this score alongside a careful examination of the Question and the Section.\n4. Key Words: Identify key words in the Section that are specifically and directly related to the Question. Such key words could include specific locations, events, or people mentioned in the Section.\n5. Background knowledge and context: Use your background knowledge of the subject matter to further elaborate on whether the Section is directly and specifically relevant to answering the Question.\n6. Relevance Determination: Based on your review of the earlier steps in the Method, determine whether the section is relevant, and gauge your confidence (high, medium, low, or none)  in this determination. High determination is specifically and directly related to the Question. If the section is relevant and ranked high, write ‘'Section_x: Relevant'. Otherwise, if the section is not relevant and the determination is less than high, write 'Section_x: Irrelevant'.\n7. Relevance Explanation: Based on your review in the earlier steps in the Method, explain why the Section’s relevance to the Question.\nLet’s begin.",
                # The suffix is some text that goes after the examples in the prompt.
                # Usually, this is where the user input will go
                suffix="Question: {question}\nKey Terms:",
                # The input variables are the variables that the overall prompt expects.
                input_variables=["question"],
                # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                example_separator="\n\n"
            )


            #running langchain prompt over 3 text sections identified previously
            llm = OpenAI(model_name="text-davinci-003", temperature=0.0)
            chain = LLMChain(llm=llm, prompt=prompt_from_string_examples)
            r_check_1 = chain.run(str(submission_text + "\n2. Section:\n " + combined1 + "\n3. SSN: " + str(similarity1)))
            r_check_2 = chain.run(str(submission_text + "\n2. Section:\n " + combined2 + "\n3. SSN: " + str(similarity2)))
            r_check_3 = chain.run(str(submission_text + "\n2. Section:\n " + combined3 + "\n3. SSN: " + str(similarity3)))

            ####combinesfunction for combining sections + outputs, and then filtering via regex for relevant sections

            st.write("Step 2 complete - relevancy check completed.")


            # combined function for combining sections + outputs, and then filtering via regex for relevant sections

            # version 1.2 - mask function for #relevance_check_w_similairities.0, only returns sections for further call as individual objects
        #combined function for combining sections + outputs, and then filtering via regex for relevant sections
        #don't delete


            # combined function for combining sections + outputs, and then filtering via regex for relevant sections

            combined_df = pd.DataFrame(columns=['output', 'r_check'])
            combined_df['output'] = [combined1, combined2, combined3]
            combined_df['r_check'] = [r_check_1, r_check_2, r_check_3]

                # Use the re.IGNORECASE flag to make the regular expression case-insensitive
            regex = re.compile(r'(section_\d+:\srelevant)', re.IGNORECASE)

                # Apply the regex pattern to the 'r_check' column and store the results in a new 'mask' column
            combined_df['mask'] = combined_df['r_check'].str.extract(regex).get(0).notnull()

                # Create a second mask to capture "this is relevant"
            combined_df['second_mask'] = combined_df['r_check'].str.contains(r'this section is relevant', flags=re.IGNORECASE)

                # Combine the two masks using the bitwise OR operator (|) and store the result in the 'mask' column
            combined_df['mask'] = combined_df['mask'] | combined_df['second_mask']

                # Filter the combined dataframe to include only rows where the 'mask' column is True
            relevant_df = combined_df.loc[combined_df['mask']].copy()

                # Check if there are any rows in the relevant_df dataframe
            if relevant_df.empty:
                # If there are no rows, print the desired message
                st.header("GPT-3 determined that none of the selected text sections are relevant to your question. Here is GPT-3's analysis of those sections.")
                st.dataframe(combined_df)
            else:
                # Otherwise, continue with the rest of the script
                def combine_strings(row):
                    return row['output'] + '\nKey Terms\n' + row['r_check']
                # Use the apply function to apply the combine_strings function to each row of the relevant_df dataframe
                # and assign the result to the 'combined_string' column
                relevant_df['combined_string'] = relevant_df.apply(combine_strings, axis=1)
                final_sections = relevant_df['combined_string']
                #final_sections.to_csv('final_sections.csv')
                evidence_df = pd.DataFrame(final_sections)
                evidence = '\n\n'.join(evidence_df['combined_string'])
                evidence_df.to_csv('evidence.csv')

                #print(evidence)

                # Filter the relevant_df dataframe to include only the 'output' column
                output_df = relevant_df[['output']]

                # Convert the dataframe to a dictionary
                output_dict = output_df.to_dict('records')

                # Extract the values from the dictionary using a list comprehension
                relevant_texts = [d['output'] for d in output_dict]

                # Print the output values to see the results
                #st.write(relevant_texts)

                    ### final answer prompt example
                    # Answer w/ Quotation - version 0

                windsor_analysis = "2. Section: Summary: Section_1:  King Edward IV was a beloved king who was interred at Windsor with great honor. He was especially beloved by the people at the time of his death. Text: Section_1: This noble prince died at his palace of Westminster and, with great funeral honor and heaviness of his people from thence conveyed, was interred at Windsor. He was a king of such governance and behavior in time of peace (for in war each part must needs be another's enemy) that there was never any prince of this land attaining the crown by battle so heartily beloved by the substance of the people, nor he himself so specially in any part of his life as at the time of his death.\n3.Initial Answer: King Edward IV was buried at Windsor with great honor and mourning from his people.\n4.Supporting Quote: ‘This noble prince died at his palace of Westminster and, with great funeral honor and heaviness of his people from thence conveyed, was interred at Windsor.’ (S.1)\n5. Combined Answer: King Edward IV was interred at Windsor with great honor and mourned by his people: ‘This noble prince...was interred at Windsor...and at the time of his death there was never any prince of this land attaining the crown by battle so heartily beloved by the substance of the people.’ (S.1)\nExcellent. Let’s try another.",
                wales_analysis = "2. Summary: Section_17:  After King Edward IV's death, his son Prince Edward moved towards London. He was accompanied by Sir Anthony Woodville, Lord Rivers, and other members of the queen's family. Text: Section_17: As soon as the King was departed, that noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales.  Such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected. And for this reason the Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages.  To the governance and ordering of this young Prince, at his sending thither, was there appointed Sir Anthony Woodville, Lord Rivers and brother unto the Queen, a right honorable man, as valiant of hand as politic in counsel. Adjoined were there unto him others of the same party, and, in effect, every one as he was nearest of kin unto the Queen was so planted next about the Prince.\n3. Initial Answer: Wales is mentioned in the text as the place where Prince Edward kept household at the time of his father's death and where he was sent to maintain order and restrain criminal activity.\n4. Supporting Quote: 'That noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales…That the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages.' (S.17)\n5. Combined Answer: Wales is mentioned in the text as the place where Prince Edward kept household and was sent to maintain order and prevent crime: 'That noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales...That the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages.' (S.17)",
                edward_question = "2. Summary: Section_2:  The people's love for King Edward IV increased after his death, as many of those who bore him grudge for deposing King Henry VI were either dead or had grown into his favor. Text: Section_2: Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly. At such time as he died, the displeasure of those that bore him grudge for King Henry's sake, the Sixth, whom he deposed, was well assuaged, and in effect quenched, in that many of them were dead in the more than twenty years of his reign a great part of a long life. And many of them in the meantime had grown into his favor, of which he was never sparing.\nInitial Answer: The public regarded Edward IV highly, with their love for him increasing after his death as many of those who bore him grudge for deposing Henry VI either died or grew into his favor.\nSupporting Quote: 'Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly...At such time as he died, the displeasure of those that bore him grudge for King Henry's sake, the Sixth, whom he deposed, was well assuaged, and in effect quenched, in that many of them were dead in the more than twenty years of his reign a great part of a long life. And many of them in the meantime had grown into his favor, of which he was never sparing.' (S.2)\nCombined Answer: The public regarded Edward IV highly at the time of his death, with their love for him increasing over time. 'Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly.' (S.2)\n. Excellent. Let’s try another."



                ### langchain final answer few-shot prompt template.
                # # Answer w/ Quotation - version 0
                examples = [
                    {"question": "Question: Where was Edward IV buried?", "output": windsor_analysis},
                    {"question": "Question: Is Wales mentioned in the text?", "output": wales_analysis},
                    {"question": "How did the public regard Edward IV?", "output": edward_question}
                ],
                    # This how we specify how the example should be formatted.
                example_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="question: {question}",
                )

                #### langchain final answer prompt structure
                #Answer w/ Quotation - version 0
                #don't delete

                prompt_from_string_examples = FewShotPromptTemplate(
                    # These are the examples we want to insert into the prompt.
                    examples=examples,
                    # This is how we want to format the examples when we insert them into the prompt.
                    example_prompt=example_prompt,
                    # The prefix is some text that goes before the examples in the prompt.
                    # Usually, this consists of intructions.
                    prefix="You are an AI question-answerer and quotation-selector. The focus of your expertise is interpreting “The History of Richard III” by Thomas More. In this exercise you will first be given a user question, a Section of More’s text, and a Method for answering the question and supporting it with an appropriate quotation from the Section. In following this Method you will complete each step by step until finished.\nHere is your Method.\nMethod: Go step by step in the question.\n1. Question: You will be provided with a user question.\n2. Section: You will be given a section from Thomas More's 'The History of Richard III.'\n3. Compose Initial Answer: Based on the Question and information provided in the Section, compose a historically accurate Initial Answer to that Question. The Initial Answer should be incisive, brief, and well-written.\n4. Identify Supporting Quote: Based on the Answer, select a Quote from the Section that supports that Answer. Be sure to only select Quotes from the “Text:Section_number” part of the Section. Select the briefest and most relevant Quote possible. You can also use paraphrasing to further shorten the Quote. Cite the Section the Quote came from, in the following manner: (S.1) for quotes from Section_1.\n5. Combined Answer with Supporting Quote: Rewrite the Initial Answer to incorporate the Quote you’ve identified from the “Text:Section_number” part of the Section. This Combined Answer should be historically accurate, and be incisive, brief, and well-written. All Quotes used should be cited using the method above.\nLet’s begin.",
                    # The suffix is some text that goes after the examples in the prompt.
                    # Usually, this is where the user input will go
                    suffix="Question: {question}\nInitial Answer:",
                    # The input variables are the variables that the overall prompt expects.
                    input_variables=["question"],
                    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                    example_separator="\n\n"
                )

                llm = OpenAI(model_name="text-davinci-003", max_tokens = 750, temperature=0.0)
                chain = LLMChain(llm=llm, prompt=prompt_from_string_examples)

                # Create an empty list to store the initial_analysis results
                initial_analysis_results = []

                # Iterate over the relevant_texts list
                for output_value in relevant_texts:
                    # Run the initial_analysis step and store the result in a variable
                    initial_analysis = chain.run(submission_text+output_value)
                    # Add the initial_analysis result to the list
                    initial_analysis_results.append(initial_analysis)

                # Create a Pandas dataframe from the relevant_texts list
                initial_analysis_df = pd.DataFrame({'relevant_texts': relevant_texts, 'initial_analysis': initial_analysis_results})
                #initial_analysis_df.to_csv('initial_analysis.csv', index=False)

                st.write("Step 3 complete - initial anaylsis finished. One final step remaining.")
                # Save the dataframe to a CSV file

                # final answer prompt, version 0 - includes biographical, context, and final answer with supporting quote.
                windsor_analysis = "2. Section: Summary: Section_1:  King Edward IV was a beloved king who was interred at Windsor with great honor. He was especially beloved by the people at the time of his death. Text: Section_1: This noble prince died at his palace of Westminster and, with great funeral honor and heaviness of his people from thence conveyed, was interred at Windsor. He was a king of such governance and behavior in time of peace (for in war each part must needs be another's enemy) that there was never any prince of this land attaining the crown by battle so heartily beloved by the substance of the people, nor he himself so specially in any part of his life as at the time of his death.\n3.Initial Answer: King Edward IV was buried at Windsor with great honor and mourning from his people.\n4.Supporting Quote: ‘This noble prince died at his palace of Westminster and, with great funeral honor and heaviness of his people from thence conveyed, was interred at Windsor.’ (S.1)\n5. Combined Answer: King Edward IV was interred at Windsor with great honor and mourned by his people: ‘This noble prince...was interred at Windsor...and at the time of his death there was never any prince of this land attaining the crown by battle so heartily beloved by the substance of the people.’ (S.1)\n6. Biographical Identification: King Edward IV\n7.Events Identification: The death and burial of King Edward IV\n8. Broader Historical Context: Windsor Castle was a popular site for royal burials in England during this time period, and the fact that King Edward IV was buried there with great honor and mourning from his people suggests that he was highly regarded by the public. The mention of his popularity at the time of his death also implies that his reign was generally well-regarded by the people of England.\n9.Final Answer: 'This noble prince died at his palace of Westminster and, with great funeral honor and heaviness of his people from thence conveyed, was interred at Windsor' (S.1). King Edward IV was buried at Windsor with great honor and mourned by his people.\nExcellent. Let’s try another.",
                wales_analysis = "2. Section & Analysis:\nSummary: Section_17:  After King Edward IV's death, his son Prince Edward moved towards London. He was accompanied by Sir Anthony Woodville, Lord Rivers, and other members of the queen's family. Text: Section_17: As soon as the King was departed, that noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales.  Such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected. And for this reason the Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages.  To the governance and ordering of this young Prince, at his sending thither, was there appointed Sir Anthony Woodville, Lord Rivers and brother unto the Queen, a right honorable man, as valiant of hand as politic in counsel. Adjoined were there unto him others of the same party, and, in effect, every one as he was nearest of kin unto the Queen was so planted next about the Prince.\n3. Initial Answer: Prince Edward was sent to Wales to keep order and restrain criminals from committing outrages.\n4. Supporting Quote: 'As soon as the King was departed, that noble Prince his son drew toward London, who at the time of his father's death kept household at Ludlow in Wales.  Such country, being far off from the law and recourse to justice, was begun to be far out of good will and had grown up wild with robbers and thieves walking at liberty uncorrected.' (S.17).\n5. Combined Answer with Supporting Quote: Prince Edward was sent to Wales to keep order and restrain criminals from committing outrages, as evidenced by the appointment of Sir Anthony Woodville, Lord Rivers and other members of the Queen's family to accompany him and act as his advisors and protectors (S.17).\n6. Biographical Identification: Prince Edward, Sir Anthony Woodville, Lord Rivers\n7. Events Identification: The death of King Edward IV, Prince Edward's move towards London, the appointment of Sir Anthony Woodville and Lord Rivers to govern and order Prince Edward in Wales.\n8. Broader Historical Context: The reign of King Edward IV was marked by political instability and conflict, particularly within the royal family. Wales, as a border region with a history of rebellion, was often used as a place to exile or restrict the movements of potential threats to the crown. In this case, it seems that Prince Edward, the son of King Edward IV, was sent to Wales to keep order and prevent further outbreaks of crime and disorder.\9.Final Answer: Wales played a role in the events mentioned in the text as a place where Prince Edward, the son of King Edward IV, was sent to maintain order and prevent crime. 'The Prince was, in the life of his father, sent thither, to the end that the authority of his presence should restrain evilly disposed persons from the boldness of their former outrages' (S.17).\nExcellent. Let’s try another.",
                edward_analysis = "2. Section:\nSummary: Section_2:  The people's love for King Edward IV increased after his death, as many of those who bore him grudge for deposing King Henry VI were either dead or had grown into his favor. Text: Section_2: Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly. At such time as he died, the displeasure of those that bore him grudge for King Henry's sake, the Sixth, whom he deposed, was well assuaged, and in effect quenched, in that many of them were dead in the more than twenty years of his reign a great part of a long life. And many of them in the meantime had grown into his favor, of which he was never sparing.\n3. Initial Answer: The public regarded Edward IV highly, with their love for him increasing after his death as many of those who bore him grudge for deposing Henry VI either died or grew into his favor.\n4. Supporting Quote: 'Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly...At such time as he died, the displeasure of those that bore him grudge for King Henry's sake, the Sixth, whom he deposed, was well assuaged, and in effect quenched, in that many of them were dead in the more than twenty years of his reign a great part of a long life. And many of them in the meantime had grown into his favor, of which he was never sparing.' (S.2)\n5. Combined Answer: The public regarded Edward IV highly at the time of his death, with their love for him increasing over time. 'Even after his death, this favor and affection toward him because of the cruelty, mischief, and trouble of the tempestuous world that followed afterwards increased more highly.' (S.2)\n6. Biographical Identification: King Edward IV, King Henry VI\n7. Events Identification: The death of King Edward IV, the deposing of King Henry VI\n8. Broader Historical Context: The reign of King Edward IV was marked by political instability and conflict, particularly with the supporters of the deposed King Henry VI. However, as time passed and many of Edward IV's opponents died or reconciled with him, the public's love for him increased. This may have been due in part to the tumultuous events that occurred after his death, which may have made the stability of his reign more attractive in retrospect.\n9. Final Answer: The public regarded Edward IV highly at the time of his death, with their love for him increasing over time as many of those who bore him grudge for deposing Henry VI either died or grew into his favor. 'At such time as he died, the displeasure of those that bore him grudge for King Henry's sake, the Sixth, whom he deposed, was well assuaged, and in effect quenched, in that many of them were dead in the more than twenty years of his reign a great part of a long life. And many of them in the meantime had grown into his favor, of which he was never sparing' (S.2).\nExcellent. Let’s try another."


                # # final answer prompt, version 0 - includes biographical, context, and final answer with supporting quote.
                examples = [
                    {"question": "Question: Where was Edward IV buried?", "output": windsor_analysis},
                    {"question": "Question: Is Wales mentioned in the text?", "output": wales_analysis},
                    {"question": "Question: How did the public regard Edward IV?", "output": edward_analysis}
                ],
                # This how we specify how the example should be formatted.
                example_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="question: {question}",
                )

                #final answer prompt, version 0 - includes biographical, context, and final answer with supporting quote.
                #don't delete

                prompt_from_string_examples = FewShotPromptTemplate(
                    # These are the examples we want to insert into the prompt.
                    examples=examples,
                    # This is how we want to format the examples when we insert them into the prompt.
                    example_prompt=example_prompt,
                    # The prefix is some text that goes before the examples in the prompt.
                    # Usually, this consists of intructions.
                    prefix="You are an AI historian with expertise in the period of the War of the Roses and the reign of Richard III of England. The focus of your expertise is interpreting “The History of Richard III” by Thomas More. In this exercise you will first be given a user question. Then you will be given a Relevant Section of the text, along with an Initial Answer, Supporting Quote, and Combined Answer with Supporting Quote. Using the Method below, provide broader historical context and academic interpretations to these Section(s) that will aid in our understanding.\nMethod. Let’s take this step by step:\n1. Question: You are first given a user-submitted question.\n2. Relevant Section & Analysis: You are then given Relevant Section from More’s 'History of Richard III.\n3. Initial Answer: You are also given an initial answer to the Question using the Relevant Section.  Please note that while the Section is likely relevant to the Question, the reasoning behind the Initial Answer may be flawed.\n4. Supporting Quote: This quote drawn from the Section supports the reasoning of the Initial Answer.\n5. Combined Answer with Supporting Quote: This is a synthesis of the Initial Answer and Supporting Quote.\n6. Biographical Identification: You work starts here. Based on the Question and Relevant Section identify individuals mentioned in the text who are relevant in answering the Question. Don't use information contained in the Initial Answer, Supporting Quote, or Combined Answer. If no individuals are mentioned, list 'None.'\n7. Events Identification: Based on the Question and the Relevant Section, identify the key events depicted in the Section and their significance to the Question. Don't use information contained in the Initial Answer, Supporting Quote, or Combined Answer.\n8. Broader Historical Context: Use your knowledge of the reign of Richard III and the Hundred Years War to analyze the Question and Section. Don't use information contained in the Initial Answer, Supporting Quote, or Combined Answer.\n9. Final Answer: Based on the information obtained in previous steps (including Initial Answer, Supporting Quote, or Combined Answer), provide a Final Answer to the Question. The Final Answer should use a quote from the Section, and cite that Quote in the style of the Supporting Quote.\nLet's begin.",
                    # The suffix is some text that goes after the examples in the prompt.
                    # Usually, this is where the user input will go
                    suffix="Question: {question}\nBiographical Information: \n\n",
                    # The input variables are the variables that the overall prompt expects.
                    input_variables=["question"],
                    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
                    example_separator="\n\n"
                )

                chain = LLMChain(llm=llm, prompt=prompt_from_string_examples)
                llm = OpenAI(model_name="text-davinci-003", max_tokens = 1000, temperature=0.0)

                #final answer prompt, version 0 - includes biographical, context, and final answer with supporting quote.
                #don't delete



                # Load the initial_analysis dataframe from the CSV file
                df = initial_analysis_df
                final_analysis = []
                relevant_texts = []
                initial_analysis = []

                # Iterate over the rows of the dataframe
                for index, row in df.iterrows():
                    # Get the values for the relevant_texts and initial_analysis columns for this row
                    output_value = row['relevant_texts']
                    analysis = row['initial_analysis']

                    # Add the values to the lists
                    relevant_texts.append(output_value)
                    initial_analysis.append(analysis)

                    # Run the initial_analysis step using the values from the dataframe
                    final_output = chain.run(submission_text+output_value+analysis)
                    final_analysis.append(final_output)
                    #print(final_output)

                # Create the final_outputs_df dataframe using the updated lists
                final_outputs_df = pd.DataFrame({'relevant_texts': relevant_texts, 'initial_analysis': initial_analysis, 'final_analysis': final_analysis})
                st.write("Step 4 completed - GPT'3 analysis is complete.")

                # Save the dataframe to a CSV file
                #final_outputs_df.to_csv('final_outputs.csv', index=False)

                # Get the number of rows in the dataframe
                num_rows = final_outputs_df.shape[0]

                if num_rows == 1:
                    st.subheader("GPT-3 has provided " + str(num_rows) + " answer to your question.")
                else:
                    st.subheader("GPT-3 has provided " + str(num_rows) + " answers to your question.")

                # Iterate through the rows of the dataframe
                for i in range(num_rows):
                  # Get the current row
                  row = final_outputs_df.iloc[i]

                  # Create an expander for the current row, with the label set to the row number
                  with st.expander(label="Answer " + str(i+1) + ":", expanded=False):
                    st.markdown("**Question:**")
                    st.write(submission_text)
                    st.markdown("**Below is GPT-3's analysis of a section of More's text that it found relevant to your qustion.**")
                    section = row['relevant_texts']
                    st.write(section)
                    #st.write(row['initial_analysis'])
                    analysis = "Biographical Identification " + row['final_analysis']
                    st.markdown(analysis)

                    def initial_output_collection():
                        now = dt.now()
                        d1 = {'question':[submission_text], 'section':[section], 'analysis':[analysis], 'date':[now]}
                        df1 = pd.DataFrame(data=d1, index=None)
                        sh1 = gc.open('aas_more_outputs')
                        wks1 = sh1[0]
                        cells1 = wks1.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                        end_row1 = len(cells1)
                        wks1.set_dataframe(df1,(end_row1+1,1), copy_head=False, extend=True)

                    initial_output_collection()

                st.header("Below is GPT-3's chain-of-thought process for generating these respones.")
                st.dataframe(final_outputs_df)








        if search_method == semantic_search:
            embeddings_search()
        if search_method == ask_a_paragraph:
            self_ask_with_search()
        else:
            ask_a_source()

def button_two():
    #Rank Bacon_bot Responses
    with col1:
        st.write("Rank the AI's Interpretation:")
        sh1 = gc.open('AAS_temp')

        wks1 = sh1[0]
        submission_text = wks1.get_value('F2')
        output = wks1.get_value('G2')
        prompt_text = wks1.get_value('D2')
        st.subheader('Your Question')
        st.write(submission_text)
        st.subheader("The AI's Answer:")
        st.write(initial_analysis)
        st.subheader("The AI's Interpretation:")

        with st.form('form2'):
            accuracy_score = st.slider("Is the AI's answer accuracte?", 0, 10, key='accuracy')
            text_score = st.slider("Are the text sections the AI selected appropriate to the question?", 0, 10, key='text')
            interpretation_score = st.slider("How effective was the AI's interpretation of the texts?", 0, 10, key='interpretation')
            coherence_rank = st.slider("How coherent and well-written is the reply?", 0,10, key='coherence')
            st.write("Transmitting the rankings takes a few moments. Thank you for your patience.")
            submit_button_2 = st.form_submit_button(label='Submit Ranking')

            if submit_button_2:
                sh1 = gc.open('AAS_outputs_temp')
                wks1 = sh1[0]
                df = wks1.get_as_df(has_header=True, index_column=None, start='A1', end=('K2'), numerize=False)
                name = df['user'][0]
                submission_text = df['question'][0]
                output = df['initial_analysis'][0]
                combined_df = df['combined_df'][0]
                relevant_texts = df['evidence'][0]
                now = dt.now()
                ranking_score = [accuracy_score, text_score, interpretation_score, coherence_rank]
                ranking_average = mean(ranking_score)

                def ranking_collection():
                    d4 = {'user':["0"], 'user_id':[user_id],'question':[submission_text], 'output':[initial_analysis], 'accuracy_score':[accuracy_score], 'text_score':[text_score],'interpretation_score':[interpretation_score], 'coherence':[coherence_rank], 'overall_ranking':[ranking_average], 'date':[now]}
                    df4 = pd.DataFrame(data=d4, index=None)
                    sh4 = gc.open('AAS_rankings')
                    wks4 = sh4[0]
                    cells4 = wks4.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                    end_row4 = len(cells4)
                    wks4.set_dataframe(df4,(end_row4+1,1), copy_head=False, extend=True)

                ranking_collection()
                st.write('Rankings recorded - thank you! Feel free to continue your conversation with Francis Bacon.')





with col2:

    st.write("Select the 'Ask Bacon' button to ask the AI questions. Select 'Rank Bacon' to note your impressions of its responses.")


    pages = {
        0 : button_one,
        1 : button_two,
    }

    if "current" not in st.session_state:

        st.session_state.current = None

    if st.button("Ask More"):
        st.session_state.current = 0
    if st.button("Rank More"):
        st.session_state.current = 1

    if st.session_state.current != None:
        pages[st.session_state.current]()
