import os
import faiss
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import ast
from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import TfidfVectorizer

model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

df = pd.read_csv("data/dresses.csv")
image_dir = "data/images"

df["selling_price_inr"] = df["selling_price"].apply(lambda x: ast.literal_eval(x)["INR"] if pd.notna(x) else None)
df["mrp_inr"] = df["mrp"].apply(lambda x: ast.literal_eval(x)["INR"] if pd.notna(x) else None)
df["launch_on"] = pd.to_datetime(df["launch_on"], errors="coerce")

brand_counts = df["brand"].value_counts()
threshold = 365

def build_text(row):
    parts = [
        str(row.get("brand", "")),
        str(row.get("product_name", "")),
        str(row["description"]) if pd.notna(row.get("description")) else "",
        str(row["meta_info"]) if pd.notna(row.get("meta_info")) else "",
        str(row["feature_list"]) if pd.notna(row.get("feature_list")) else "",
        str(row["style_attributes"]) if pd.notna(row.get("style_attributes")) else "",
    ]
    return " ".join(parts).lower()

df["text_description"] = df.apply(build_text, axis=1)

faiss_path = "data/faiss_index.bin"
meta_path = "data/metadata_balanced.npy"

if os.path.exists(faiss_path) and os.path.exists(meta_path):
    index = faiss.read_index(faiss_path)
    metadata = np.load(meta_path, allow_pickle=True).tolist()
else:
    index = faiss.IndexFlatL2(512)
    embeddings = []
    metadata = []

    for _, row in df.iterrows():
        brand = row["brand"]
        pid = row["product_id"]
        base_meta = {
            "product_id": pid,
            "product_name": row.get("product_name", ""),
            "brand": brand,
            "text_description": row["text_description"],
            "launch_on": row["launch_on"],
        }

        image_paths = []
        # Oversampled → only main
        if brand_counts[brand] > threshold:
            image_paths = [row["feature_image_s3"]]
        else:
            # Use main + 1 additional pdp image
            pdp_list = ast.literal_eval(row["pdp_images_s3"]) if pd.notna(row["pdp_images_s3"]) else []
            image_paths = [row["feature_image_s3"]] + pdp_list[:1]

        for i, img_url in enumerate(image_paths):
            img_filename = f"{pid}_{i}.jpg"
            img_path = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_path):
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    feat = model.get_image_features(**inputs).cpu().numpy()[0]
                embeddings.append(feat)
                metadata.append({**base_meta, "image_path": img_path})
            except Exception as e:
                print(f"❌ Skipped {img_path}: {e}")

    index.add(np.stack(embeddings))
    faiss.write_index(index, faiss_path)
    np.save(meta_path, metadata)

def get_similar_dresses(query_image, k=5, return_top_match=False):
    inputs = processor(images=query_image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)[0].cpu().numpy().reshape(1, -1)
    D, I = index.search(query_emb.astype("float32"), k)
    results = [metadata[i] for i in I[0]]
    pid = results[0]["product_id"] if return_top_match else None
    return (results, pid) if return_top_match else results

def get_personalized_recommendations(history_pids, k=5):
    if not history_pids:
        return []
    past_rows = [row for row in metadata if row["product_id"] in history_pids]
    texts = [r["text_description"] for r in past_rows if "text_description" in r]
    vectorizer = TfidfVectorizer(max_features=200)
    tfidf_matrix = vectorizer.fit_transform(texts)
    avg_vector = tfidf_matrix.mean(axis=0)
    results = []
    for row in metadata:
        if row["product_id"] in history_pids:
            continue
        if "text_description" not in row:
            continue
        candidate = vectorizer.transform([row["text_description"]])
        sim = float((avg_vector @ candidate.T)[0, 0])
        results.append((sim, row))
    return [row for _, row in sorted(results, key=lambda x: x[0], reverse=True)[:k]]

def get_trending_recommendations(matched_pid, k=5):
    ref = next((r for r in metadata if r["product_id"] == matched_pid), None)
    if not ref:
        return []

    keywords = set(re.findall(r"\b\w+\b", ref["text_description"]))
    recent_cutoff = datetime.now() - timedelta(days=90)
    candidates = []
    for row in metadata:
        if row["product_id"] == matched_pid:
            continue
        if pd.isna(row["launch_on"]) or row["launch_on"] < recent_cutoff:
            continue
        row_text = row.get("text_description", "")
        if row["brand"] == ref["brand"] or any(k in row_text for k in keywords):
            candidates.append(row)

    # Sort recent first
    return sorted(candidates, key=lambda r: r["launch_on"], reverse=True)[:k]
