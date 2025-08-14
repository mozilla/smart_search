## To execute the static embedder
To download the embeddings file from HF
```
curl -L -o embeddings_dim_256.bin \
  https://huggingface.co/Mozilla/static-retrieval-mrl-en-v1/resolve/main/embeddings_dim_256.bin
```

In main.cpp change the downloaded file path to the correct path

```
cd static_embedder
g++ -std=c++17 -o static_embedder main.cpp tokenizer.cpp Log.cpp
./static_embedder
```