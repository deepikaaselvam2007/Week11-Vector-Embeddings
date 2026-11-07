# Week 11 – Vector Embeddings with ChromaDB

## Aim
Generate text embeddings with `all-MiniLM-L6-v2`, store documents in ChromaDB, and retrieve the Top-5 semantically similar documents for each test query.

## Folder contents
- `dataset/documents/`: 10 sample AI and education documents.
- `dataset/queries.csv`: 10 test queries.
- `vector_store.py`: creates embeddings, stores them in ChromaDB, and exports retrieval results.
- `notebook/Week11_Vector_Embeddings.ipynb`: notebook launcher.
- `chroma_db/`: created automatically after running the program.

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the `Week11_Vector_Embeddings` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create an environment: `py -3.12 -m venv .venv`
5. Activate it: `.venv\Scripts\activate`
6. Install packages: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`
7. Run: `python vector_store.py`
8. Open the notebook folder and run `Week11_Vector_Embeddings.ipynb` if you need notebook screenshots.

If `py -3.12` is unavailable, install Python 3.12 first. Python 3.12 is recommended for broad compatibility with ML packages; you can try your installed Python version, but package wheels may vary.

## Expected output
- A persistent ChromaDB collection named `ai_education_documents`.
- `dataset/embeddings.npy`
- `dataset/retrieved_results.csv` containing 50 rows (Top-5 × 10 queries).

The first run may download the embedding model, so an internet connection is required.
