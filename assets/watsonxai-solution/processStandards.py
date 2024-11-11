#preprocess the security requirements with metadata 
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker import HierarchicalChunker

import re, json
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
load_dotenv()

default_ef = embedding_functions.DefaultEmbeddingFunction()
READ_LOCAL_VDB = True

def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.environ['API_KEY']
	}

def create_standards_db() :
    #create the vector database
    vdb = configure_chroma()

    #Read the chunks
    chunks = {}
    with open("../data/standards.json") as f :
        #Open the json file and read it into the chunks dictionary
        chunks = json.load(f)

    #Add the documents to the vdb
    texts = []
    metadatas = []
    for chunk in chunks :
        texts.append(chunk["content"])
        metadatas.append({"File Name": chunk["filename"], "Heading": str(chunk["heading"])})
    ids = [f"id{i}" for i in range(len(texts))]
    vdb.add(documents=texts, ids=ids, metadatas=metadatas)

    return vdb

#Configure the chromadb, if it does not already exist
#Returns the vdb object
def configure_chroma():
    print("Loading Chroma from local.")
    chroma_client = chromadb.PersistentClient(path="../data/chromadb/")
    return chroma_client.get_or_create_collection(name="standards-rag-collection",
                                                    embedding_function= default_ef,
                                                    metadata={"hnsw:space": "cosine"})

def chunk_standards(filename) :
     # previous `PipelineOptions` is now `PdfPipelineOptions`
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    #...

    ## Custom options are now defined per format.
    doc_converter = DocumentConverter()

    conv_result = doc_converter.convert("../data/USMITSecurityStandards.pdf") # previously `convert_single`

    doc = conv_result.document
    doc_chunks = list(c.export_json_dict() for c in HierarchicalChunker().chunk(doc))

    chunks = []
    for c in doc_chunks :
        if c['meta']["headings"][0] != "Version 5.0" :
            chunks.append({"filename": c['meta']["origin"]["filename"], "content": c["text"], "heading": c['meta']["headings"][0]})


    with open('../data/standards.json', 'w') as convert_file:
        convert_file.write(json.dumps(chunks))
     

# Call the chunk_standards("Insert PDF Name") Function 
#chunk_standards("../data/USMITSecurityStandards.pdf")
