from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

class DocumentSorter():
    """
    Sorts documents into information, garbage, and questions
    """
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a system that is used to quantify chunks of text. You must only respond with one of three words: "information", "garbage", or "question".
        You will respond with "information" if the text contains information with context that can be used to answer questions.
        You will respond with "garbage" if the text contains not contextually relevant information and cannot be used to answer questions
        You will respond with "question" if the text contains a question but not the information to answer the question.
        """),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    def __init__(self):
        #Lists of information, garbage, and questions
        self.info = []
        self.garbage = []
        self.questions = []
        self.broken = []

    def sort(self, documents):
        """
        Sorts documents into information, garbage, and questions
        """
        for doc in tqdm(documents):
            response = DocumentSorter.chain.invoke({"input": doc.text})
            if response == "information":
                self.info.append(doc)
            elif response == "garbage":
                self.garbage.append(doc)
            elif response == "question":
                self.questions.append(doc)
            else:
                self.broken.append(doc)
        return self.info, self.questions, self.garbage, self.broken
    