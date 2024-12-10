# smart_search
improving search


Step 1 (Set up environment):
```
git clone https://github.com/mozilla/smart_search.git

Create python env:
if mac (m1)
brew install python3
export LDFLAGS="-L/opt/homebrew/opt/sqlite/lib"
export CPPFLAGS="-I/opt/homebrew/opt/sqlite/include"

/opt/homebrew/bin/python3 -venv venv
source venv/bin/activate 
python -m pip install -r requirements
```

Step 2 - Create Embeddings for your search history
Please follow notebook -
`notebooks/explore_semantic_search.ipynb`
Once executed move to the next step

Step 3 - if you want to run demo app (depends on step 2)
`streamlit run src/history_search_app.py`
open: `http://localhost:8501/`