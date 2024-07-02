import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def get_gpt():
	azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE") or ""
	api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
	api_version: str = os.getenv("AZURE_OPENAI_API_VERSION") or ""
	azure_openai_deployment : str = os.getenv("AZURE_OPENAI_MODEL_NAME") or ""
	llm = AzureChatOpenAI(azure_deployment=azure_openai_deployment, temperature=0, streaming=True, azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
	return llm

def get_embedding(embedding_model_name : str):
	azure_endpoint: str = os.getenv("AZURE_OPENAI_BASE") or ""
	api_key = os.getenv("AZURE_OPENAI_KEY") or ""
	azure_embedding_deployment: str = os.getenv(embedding_model_name) or ""
	embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, api_key=api_key, azure_deployment=azure_embedding_deployment)
	return embeddings

def summarize_text(content: str) -> str:
	from langchain_core.output_parsers import StrOutputParser
	from langchain.prompts import PromptTemplate
	llm = get_gpt()
	prompt_template = """Dato il seguente documento delimitato da `````, produci un riassunto dettagliato del contenuto in 30 righe
Non aggiungere commenti oltre al riassunto stesso

Documento:
`````
{documento}
`````

Riassunto:
"""
	prompt = PromptTemplate(template=prompt_template, input_variables=["documento"])
	chain = prompt | llm | StrOutputParser()
	summary = chain.invoke({"documento": content})
	return summary