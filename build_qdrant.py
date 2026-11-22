from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PointIdsList
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import uuid
from tqdm import tqdm
import os
import time
import tiktoken
from azure_blob_utils import download_blob

def update_qdrant():
    # ------------------------ Config ------------------------
    load_dotenv()
    TOKEN_LIMIT_PER_MINUTE = os.getenv("TOKEN_LIMIT_PER_MINUTE", "750000")
    BATCH_SIZE = os.getenv("BATCH_SIZE", 10)
    SLEEP_TIME = os.getenv("SLEEP_TIME", 60)

    TOKEN_LIMIT_PER_MINUTE = int(TOKEN_LIMIT_PER_MINUTE)
    BATCH_SIZE = int(BATCH_SIZE)
    SLEEP_TIME = int(SLEEP_TIME)
    # --------------------- Embedding Setup -------------------
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        azure_endpoint=os.getenv("OPENAI_EMBEDDINGS_ENDPOINT"),
        api_key=os.getenv("OPENAI_EMBEDDINGS_API"),
        api_version="2024-02-01"
    )
    tokenizer = tiktoken.get_encoding("cl100k_base")


    # ----------------------- Data Load -----------------------
    data = "posts.csv"
    azure_container = "csv-file"
    download_blob(azure_container, data, data)
    posts_df = pd.read_csv(data)
    #posts_df = posts_df.head(1000)

    documents = [
        Document(
            page_content=f"{str(row['title']).strip()} {str(row['content']).strip()}",
            metadata={"date": row["date"], "author": row["author"], "link": row["link"]}
        )
        for _, row in posts_df.iterrows()
    ]


    # ---------------------- Text Split -----------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)

    # Deterministic IDs: the same chunk of the same article always maps to the same point,
    # so re-running upserts in place instead of adding another copy of every article.
    seen = set()
    texts, metadatas, splits_id = [], [], []
    for doc in splits:
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.metadata['link']}\n{doc.page_content}"))
        if pid in seen:
            continue
        seen.add(pid)
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)
        splits_id.append(pid)
    print(f"{len(documents)} articles -> {len(texts)} unique chunks")

    # -------------------- Qdrant Setup -----------------------
    # Same connection settings as LCI-Chatbot (services/vector.py); the server requires an API key.
    qdrant_client = QdrantClient(host=os.environ["QDRANT_HOST"],
                                 port=int(os.environ.get("QDRANT_PORT", 6333)),
                                 api_key=os.environ["QDRANT_API"],
                                 https=False)
    collection_name = "articles"
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)

        )

    def all_point_ids(client, collection_name):
        ids, offset = set(), None
        while True:
            points, offset = client.scroll(collection_name=collection_name, limit=1000, offset=offset,
                                           with_payload=False, with_vectors=False)
            ids.update(str(p.id) for p in points)
            if offset is None:
                return ids

    existing_ids = all_point_ids(qdrant_client, collection_name)
    current_ids = set(splits_id)
    todo = [i for i, pid in enumerate(splits_id) if pid not in existing_ids]
    print(f"Qdrant has {len(existing_ids)} points; {len(texts) - len(todo)} chunks already present, {len(todo)} to embed")
    texts = [texts[i] for i in todo]
    metadatas = [metadatas[i] for i in todo]
    splits_id = [splits_id[i] for i in todo]


    # ------------------- Rate Limiting Utils -------------------

    def get_batches(texts, metadatas, ids, tokenizer, token_limit, batch_size):
        i = 0
        while i < len(texts):
            token_accumulator = 0
            batch_texts, batch_metadatas, batch_ids = [], [], []

            while i < len(texts) and len(batch_texts) < batch_size:
                text = texts[i]
                token_count = len(tokenizer.encode(text))

                if token_accumulator + token_count > token_limit:
                    break

                batch_texts.append(text)
                batch_metadatas.append(metadatas[i])
                batch_ids.append(ids[i])
                token_accumulator += token_count
                i += 1

            yield batch_texts, batch_metadatas, batch_ids, token_accumulator

    def upload_batch_to_qdrant(client, collection_name, texts, metadatas, ids, embeddings):
        points = [
            PointStruct(id=ids[j], vector=embeddings[j], 
                        payload = {
                            **metadatas[j],
                            "page_content": texts[j]
                        }
            )
            for j in range(len(texts))
        ]
        #print(len(points[0].vector))
        client.upsert(collection_name=collection_name, points=points)

    # ------------------- Main Processing Loop -------------------
    p_bar = tqdm(total=len(texts))
    for batch_texts, batch_metadatas, batch_ids, token_acc in get_batches(
        texts, metadatas, splits_id, tokenizer, TOKEN_LIMIT_PER_MINUTE, BATCH_SIZE
    ):
        batch_embeddings = embeddings.embed_documents(batch_texts)
        upload_batch_to_qdrant(qdrant_client, collection_name, batch_texts, batch_metadatas, batch_ids, batch_embeddings)
        p_bar.update(len(batch_texts))

        if token_acc >= TOKEN_LIMIT_PER_MINUTE:
            print(f"Hit {token_acc} tokens. Sleeping for {SLEEP_TIME} seconds...")
            with tqdm(total=SLEEP_TIME, desc="Sleeping", unit="s") as sleep_pbar:
                for _ in range(SLEEP_TIME):
                    time.sleep(1)
                    sleep_pbar.update(1)

    p_bar.close()

    # ------------------- Prune stale / duplicate points -------------------
    # Runs only after every current chunk is uploaded, so the collection is never left empty.
    # Removes old random-ID duplicates and chunks of articles that were edited or deleted.
    min_chunks = int(os.getenv("PRUNE_MIN_CHUNKS", 1000))
    if len(current_ids) < min_chunks:
        print(f"Skipping prune: only {len(current_ids)} current chunks (< {min_chunks}); posts.csv may be incomplete")
        return
    stale = [pid for pid in all_point_ids(qdrant_client, collection_name) if pid not in current_ids]
    for i in range(0, len(stale), 1000):
        qdrant_client.delete(collection_name=collection_name,
                             points_selector=PointIdsList(points=stale[i:i + 1000]))
    print(f"Pruned {len(stale)} stale/duplicate points; collection now holds {len(current_ids)} chunks")