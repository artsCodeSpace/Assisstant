# Execution Procedure

1. Create a project folder
2. Create a virtual environment
    > python -m venv .venv
3. Activate virtual environment
    > .venv\Scripts\activate
4. Update .env file with groq-apikey
5. Install the requirements.txt file
    > pip install -r requirements.txt
6. Execute the ingestion file for RAG
    > python ingest.py
7. Run main app
    > python app.py
8. Check the url in browser 
    localhost://5000