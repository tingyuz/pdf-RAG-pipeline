# pdf-RAG-pipeline

The program will load pdf file from specific folder, extract and chunk the text in a semantic-aware approach>

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

3. python pdf-rag-pipeline.py /path/to/pdf/folder


4. It is suggested to use venv to run the program.


