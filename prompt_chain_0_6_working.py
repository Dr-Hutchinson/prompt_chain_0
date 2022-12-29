from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains import LLMChain
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


scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

gc = pygsheets.authorize(custom_credentials=credentials)

#pygsheets credentials for Google Sheets API


st.set_page_config(
    page_title="Ask a Source: More's History of Richard III",
    layout='wide',
    page_icon='🔍'
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Ask A Source: Thomas More's 'The History of Richard III'")
col1, col2 = st.columns([3.0,3.5])
with col1:
    book_pic = st.image(image ='./more_page.jpg', caption="From Thomas More's 'History of Richard III' (1557). British Library.", width=500)
    #st.write("Explore the current data.")
    #df = pd.read_csv('richardbot1_data.csv')
    #st.dataframe(df, height=500)

def button_one():
    st.write("This application uses GPT-3 to answer questions about Thomas More's [_History of King Richard III_](https://thomasmorestudies.org/wp-content/uploads/2020/09/Richard.pdf). Choose one of the options below, and pose a question about the text.")

    semantic_search = "Semantic Search: Enter a question, and recieve sections of the text that are the most closely related."
    ask_a_source = "Ask A Source: Pose a question about the text, and GPT-3 will share answers drawn from the text along with historical analysis."

    search_method = st.radio("Choose a method:", (semantic_search, ask_a_source))
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


        if search_method == semantic_search:
            embeddings_search()
        else:
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
                st.header("GPT-3 determined that none of the selected text sections are relevant to your question. Here is GPT-3's1f analysis of those sections.")
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
                  with st.expander(label="Answer " + str(i) + ":", expanded=False):
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

    if st.button("Ask Bacon"):
        st.session_state.current = 0
    if st.button("Rank Bacon"):
        st.session_state.current = 1

    if st.session_state.current != None:
        pages[st.session_state.current]()
