import pandas as pd
import os
import json
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer
import time
import logging

nltk.download('punkt', quiet=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class WebAutoSearch:
    def __init__(self, data_path="cars_2010_2020.csv"):
        self.data_path = data_path
        self.encoder = None
        self.retriever = None
        self.initialized = False

    def initialize(self):
        """Initialize all components for web service"""
        try:
            logger.info("Initializing web search system...")
            self.encoder = SemanticEncoder()

            logger.info("Preparing data...")
            corpus, _, doc_id_map, df = prepare_data(self.data_path)

            logger.info("Building retriever...")
            self.retriever = AutomobileRetriever(corpus, self.encoder, doc_id_map, df)

            self.initialized = True
            logger.info("System initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def search(self, query):
        """Web-friendly search interface"""
        if not self.initialized:
            raise RuntimeError("System not initialized")

        try:
            start_time = time.time()
            results = self.retriever.search(query, top_k=10, verbose=False)

            formatted_results = []
            for car_id, score, car_info in results:
                result_data = self._format_car_info(car_info)
                result_data.update({
                    "car_id": car_id,
                    "score": round(score, 3)
                })
                formatted_results.append(result_data)

            logger.info(f"Search completed in {time.time() - start_time:.2f}s")
            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def _format_car_info(self, car_info):
        """Format car info for web response"""
        car_dict = car_info.to_dict()
        return {
            "make": self._get_attribute(car_dict, ['make', 'Make', 'MAKE', 'brand', 'Brand']),
            "model": self._get_attribute(car_dict, ['model', 'Model', 'MODEL']),
            "year": self._get_attribute(car_dict, ['year', 'Year', 'YEAR']),
            "price": self._get_attribute(car_dict, ['Price (USD)', 'price', 'Price', 'PRICE', 'msrp', 'MSRP']),
            "features": self._get_features(car_dict)
        }

    def _get_attribute(self, car_dict, possible_keys):
        """Helper to get first matching attribute"""
        for key in possible_keys:
            if key in car_dict and pd.notna(car_dict[key]):
                return str(car_dict[key])
        return "N/A"

    def _get_features(self, car_dict):
        """Extract important features"""
        features = []
        feature_map = [
            ('transmission', ['transmission', 'Transmission', 'TRANSMISSION']),
            ('engine', ['engine', 'Engine', 'ENGINE', 'engine_type', 'Engine Type']),
            ('fuel', ['fuel_type', 'Fuel Type', 'fuel', 'Fuel', 'mpg', 'MPG']),
            ('body', ['body_type', 'Body Type', 'body', 'Body']),
            ('drive', ['drive', 'Drive', 'drivetrain', 'Drivetrain']),
            ('horsepower', ['horsepower', 'Horsepower', 'hp', 'HP']),
        ]

        for name, keys in feature_map:
            value = self._get_attribute(car_dict, keys)
            if value != "N/A":
                features.append(f"{name.title()}: {value}")

        return features[:4]


# Core functionality remains unchanged below
# =================================================================

def preprocess_text(text):
    """Normalize text while preserving automotive terminology"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s.-]', '', text)
    return text.strip()


def combine_car_features(row):
    """Combine relevant car features into a coherent description"""
    features = []
    for col in row.index:
        if pd.notna(row[col]) and row[col] != "" and col not in ['index']:
            features.append(f"{col}: {row[col]}")
    return " ".join(features)


def split_into_passages(text, max_length=150):
    """Split car descriptions into coherent passages"""
    sentences = sent_tokenize(text)
    passages = []
    current_passage = []
    current_length = 0

    for sent in sentences:
        sent_length = len(sent.split())
        if current_length + sent_length <= max_length:
            current_passage.append(sent)
            current_length += sent_length
        else:
            passages.append(" ".join(current_passage))
            current_passage = [sent]
            current_length = sent_length
    if current_passage:
        passages.append(" ".join(current_passage))
    return passages


def prepare_data(data_path):
    """Prepare automotive dataset for retrieval"""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    logger.info(f"Columns in the dataset: {df.columns.tolist()}")
    logger.info(f"First row sample: {df.iloc[0].to_dict()}")

    df['description'] = df.apply(combine_car_features, axis=1)

    corpus_passages = {}
    doc_id_map = {}

    for idx, row in tqdm(df.iterrows(), desc="Processing car entries", total=len(df)):
        car_id = idx
        text = preprocess_text(row['description'])
        passages = split_into_passages(text)

        for i, passage in enumerate(passages):
            passage_key = f"car_{car_id}_{i}"
            corpus_passages[passage_key] = passage
            doc_id_map[passage_key] = car_id

    return corpus_passages, None, doc_id_map, df  # Removed unused queries_dict


class SemanticEncoder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.cache = {}

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        uncached_texts = []
        uncached_indices = []
        embeddings = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)

        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=128, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9)

                batch_np = batch_embeddings.cpu().numpy()
                for j, (text_idx, text) in enumerate(zip(uncached_indices[i:i + batch_size], batch)):
                    self.cache[text] = batch_np[j]
                    embeddings[uncached_indices[i + j]] = batch_np[j]

        return embeddings


def build_parallel_index(corpus_passages, encoder, index_path="automotive_index", use_cache=True):
    logger.info("Building parallel index...")
    passage_ids = list(corpus_passages.keys())
    passage_texts = [corpus_passages[pid] for pid in passage_ids]

    cache_file = f"{index_path}_cache.npz"

    if use_cache and os.path.exists(cache_file):
        logger.info("Loading embeddings from cache...")
        cached = np.load(cache_file)
        embeddings = cached['embeddings']
    else:
        logger.info("Encoding passages...")
        embeddings = encoder.encode(passage_texts)
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
        np.savez(cache_file, embeddings=embeddings)

    logger.info(f"Building index with {len(embeddings)} vectors of dimension {embeddings.shape[1]}...")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    n_cells = min(int(np.sqrt(len(embeddings))), 1024)
    n_cells = max(n_cells, 8)

    index = faiss.IndexIVFPQ(quantizer, d, n_cells, 16, 8)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(n_cells // 4, 32)

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
    faiss.write_index(index, index_path)

    with open(f"{index_path}_ids.json", "w") as f:
        json.dump(passage_ids, f)

    return index, passage_ids


def multi_vector_coalescing(index, passage_ids, corpus_passages, threshold=0.85):
    logger.info("Performing multi-vector coalescing...")
    n = index.ntotal
    index.make_direct_map()
    embeddings = np.vstack([index.reconstruct(i) for i in range(n)])

    doc_groups = {}
    for i, pid in enumerate(passage_ids):
        doc_id = pid.split('_')[1]
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append((i, pid))

    new_embeddings = []
    new_ids = []
    id_mapping = {}

    for doc_id, passages in doc_groups.items():
        if len(passages) == 1:
            idx, pid = passages[0]
            new_embeddings.append(embeddings[idx])
            new_ids.append(pid)
            id_mapping[pid] = len(new_ids) - 1
            continue

        passages.sort(key=lambda x: int(x[1].split('_')[2]))
        current_group = [passages[0]]

        for i in range(1, len(passages)):
            idx, pid = passages[i]
            prev_idx = current_group[-1][0]
            similarity = np.dot(embeddings[idx], embeddings[prev_idx])

            if similarity >= threshold:
                current_group.append((idx, pid))
            else:
                group_indices = [g[0] for g in current_group]
                group_ids = [g[1] for g in current_group]
                weights = np.array([len(corpus_passages[pid].split()) for pid in group_ids])
                weights /= weights.sum()

                coalesced = np.average(embeddings[group_indices], axis=0, weights=weights)
                faiss.normalize_L2(coalesced.reshape(1, -1))

                new_embeddings.append(coalesced)
                new_ids.append(group_ids[0])
                for orig_id in group_ids:
                    id_mapping[orig_id] = len(new_ids) - 1
                current_group = [(idx, pid)]

        if current_group:
            group_indices = [g[0] for g in current_group]
            group_ids = [g[1] for g in current_group]
            weights = np.array([len(corpus_passages[pid].split()) for pid in group_ids])
            weights /= weights.sum()

            coalesced = np.average(embeddings[group_indices], axis=0, weights=weights)
            faiss.normalize_L2(coalesced.reshape(1, -1))

            new_embeddings.append(coalesced)
            new_ids.append(group_ids[0])
            for orig_id in group_ids:
                id_mapping[orig_id] = len(new_ids) - 1

    new_embeddings = np.array(new_embeddings).astype(np.float32)
    logger.info(f"Coalesced {len(embeddings)} vectors to {len(new_embeddings)} vectors")

    d = new_embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    n_cells = min(int(np.sqrt(len(new_embeddings))), 1024)
    n_cells = max(n_cells, 8)

    new_index = faiss.IndexIVFPQ(quantizer, d, n_cells, 16, 8)
    new_index.train(new_embeddings)
    new_index.add(new_embeddings)
    new_index.nprobe = min(n_cells // 4, 32)

    return new_index, new_ids, id_mapping


class AutomobileRetriever:
    def __init__(self, corpus, encoder, doc_id_map, dataframe, use_cache=True):
        self.corpus = corpus
        self.encoder = encoder
        self.doc_id_map = doc_id_map
        self.dataframe = dataframe

        start_time = time.time()
        logger.info("Building dense index...")
        self.faiss_index, self.passage_ids = build_parallel_index(corpus, encoder, use_cache=use_cache)

        logger.info("Applying multi-vector coalescing...")
        self.coalesced_index, self.coalesced_ids, self.id_mapping = multi_vector_coalescing(
            self.faiss_index, self.passage_ids, self.corpus
        )

        logger.info("Building sparse index...")
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2)
        )
        passage_texts = [self.corpus[pid] for pid in self.coalesced_ids]
        self.tfidf_matrix = self.tfidf.fit_transform(passage_texts)

        logger.info(f"Index building completed in {time.time() - start_time:.2f} seconds")

    def retrieve_sparse(self, query, top_k=100):
        query_tfidf = self.tfidf.transform([query])
        scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.coalesced_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def retrieve_dense(self, query, top_k=100):
        query_embedding = self.encoder.encode([query])[0].reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        D, I = self.coalesced_index.search(query_embedding, top_k)
        return [(self.coalesced_ids[i], float(s)) for i, s in zip(I[0], D[0]) if i >= 0]

    def search(self, query, top_k=10, alpha=0.7, verbose=False):
        start_time = time.time()
        query = preprocess_text(query)

        # Sparse and dense retrieval
        dense_results = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_results = self.retrieve_sparse(query, top_k=top_k * 2)

        # Result combination
        all_ids = set([r[0] for r in dense_results] + [r[0] for r in sparse_results])
        dense_dict = {pid: score for pid, score in dense_results}
        sparse_dict = {pid: score for pid, score in sparse_results}

        max_dense = max(dense_dict.values()) if dense_dict else 1
        max_sparse = max(sparse_dict.values()) if sparse_dict else 1

        combined = {}
        for pid in all_ids:
            dense_norm = dense_dict.get(pid, 0) / max_dense
            sparse_norm = sparse_dict.get(pid, 0) / max_sparse
            combined[pid] = alpha * dense_norm + (1 - alpha) * sparse_norm

        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        car_results = []
        for pid, score in results:
            car_id = int(self.doc_id_map[pid])
            car_info = self.dataframe.iloc[car_id]
            car_results.append((car_id, score, car_info))

        return car_results


if __name__ == "__main__":
    # Example usage
    search_system = WebAutoSearch()
    search_system.initialize()
    results = search_system.search("SUV with good fuel economy")
    print(json.dumps(results[:3], indent=2))