#  Fashion Visual Similarity Search App

A fully containerized, visual search and recommendation app for fashion items. Upload a dress image and get:
-  Visually similar products
-  Personalized recommendations (based on your search history)
- 🔥 Trending similar dresses (recent launches that match your style)

---

## 🚀 Features

### ✅ Outcome 1: Visual Similarity (CLIP + FAISS)
- Uses FashionCLIP to embed product and query images
- Retrieves top 5 visually similar dresses from pre-indexed embeddings

### ✅ Outcome 2: Personalized Recommendations
- Tracks your uploaded image history
- Extracts dominant features (color, style, cloth material, brand)
- Recommends based on recurring patterns in your search behavior

### ✅ Outcome 3: Trending Similar Dresses
- Matches uploaded image against dataset by style and brand
- Filters for dresses launched in the last 90 days
- Returns trending products of similar type

---

## 🛠️ Why We Did What We Did

| Challenge                        | Solution                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| Imbalanced brand representation  | Hybrid sampling: undersample oversampled brands, enhance rare ones      |
| Missing data in metadata         | Constructed robust `text_description` using available fields             |
| Slow/unstable embedding          | Batched embedding + caching with FAISS and `.npy` metadata               |
| Redundant image usage            | Embedded fewer images for oversampled brands, more for underrepresented |
| Dynamic personalization          | TF-IDF similarity on history-based descriptions                         |
| Trend logic too strict           | Loosened filters + fallback to most recent similar dresses               |

---

## 📦 How to Run with Docker

```bash
docker build -t fashion-search .
docker run -p 8501:8501 fashion-search
