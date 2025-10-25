# pdf-RAG-pipeline

The program will load pdf files from a specific folder, extract and chunk the text in a semantic-aware approach, then use it with llm to answer questions related to pdf files.

1.
```
pip install faiss-cpu PyMuPDF numpy nltk ollama
```

2.
```
import nltk
nltk.download('punkt')
nltk.download('punkt-tab')
```

3. 
```
python rag-pipeline.py /path/to/pdf/folder
```

4. It is suggested to use venv to run the program.


