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

/opt/homebrew/bin/python3 -m venv venv
source venv/bin/activate 
python -m pip install -r requirements.txt
```

Step 2 - Create Embeddings for your search history
```
Please follow notebook => notebooks/explore_semantic_search.ipynb
Once executed move to the next step
```

Step 3 - if you want to run demo app (depends on step 2)
```
streamlit run src/history_search_app.py
open http://localhost:8501/
```

#####################################################################
Instructions on the KG 

Pre-req:
```
source venv/bin/activate
python -m pip install -r requirements.txt
copy places.sqlite to data/places.sqlite
```

Step 1) To Build KG database
```
Note: For the first time edit GENERATE_TOPIC = True and next time onwards flip to False
python src/kg_builder.py
```

Step 2) To validate KG approach
```
## Override with your golden queries (if does not exist, then uses moz_inputhistory table)
GOLDEN_QUERIES_FILE = f"{DATA_PATH}/chidam_golden_query.csv"

## you could also change the row_limit = 10000 to smaller number 
python src/kg_validator.py
```