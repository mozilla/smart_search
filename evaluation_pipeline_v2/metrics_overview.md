## Evaluation Metrics Overview

This section explains the information retrieval (IR) metrics used in our evaluation pipeline.  
All metrics are computed **per query**, and aggregate results are reported as **macro averages** (mean across queries).

---

### 1. Precision@K
**Definition:**  
Precision@K measures how many of the top-K retrieved documents are relevant.

$$
\text{Precision@K} = \frac{|\text{Retrieved}_{K} \cap \text{Relevant}|}{K}
$$

**Interpretation:**  
Higher Precision@K means the top-ranked documents are mostly relevant.  
It focuses on the **quality** of the retrieved set rather than completeness.

---

### 2. Recall@K
**Definition:**  
Recall@K measures how many of the relevant documents were successfully retrieved within the top-K results.

$$
\text{Recall@K} = \frac{|\text{Retrieved}_{K} \cap \text{Relevant}|}{|\text{Relevant}|}
$$

**Interpretation:**  
Higher Recall@K means the system is better at covering all relevant results, even if some irrelevant items are also retrieved.

---

### 3. NDCG@K (Normalized Discounted Cumulative Gain)
**Definition:**  
NDCG@K rewards relevant documents appearing **earlier** in the ranking.  
The DCG (Discounted Cumulative Gain) is normalized by the ideal DCG (IDCG).

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)} \qquad
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

where \( rel_i = 1 \) if the document at rank *i* is relevant, else 0.

**Interpretation:**  
NDCG emphasizes correct **ordering**, retrieving relevant documents at higher ranks yields higher scores.

---

### 4. Reciprocal Rank (RR)
**Definition:**  
The reciprocal rank is the inverse of the rank position of the first relevant document.

$$
\text{RR} = \frac{1}{\text{rank of first relevant document}}
$$

**Interpretation:**  
If the first relevant document appears at rank 1, RR = 1.0;  
if it appears at rank 5, RR = 0.2.  
Average RR across queries gives the **Mean Reciprocal Rank (MRR)**, a measure of how early the first correct answer is found.

---

### 5. Average Precision (AP)
**Definition:**  
Average Precision computes the mean of all Precision@K values where a relevant document is retrieved.

$$
\text{AP} = \frac{1}{|\text{Relevant}|} \sum_{k=1}^{N} P(k) \cdot rel(k)
$$

where \( P(k) \) is the precision at cutoff *k*, and \( rel(k) \) = 1 if the document at rank *k* is relevant.

**Interpretation:**  
AP accounts for both **precision** and **recall**, giving higher weight to relevant items retrieved earlier.  
The mean of AP over all queries is the **Mean Average Precision (MAP)**.

---


### Summary of Metric Roles

| Metric                      | Focus | Intuition |
|:----------------------------|:--|:--|
| **Precision@K**             | Quality | Are the top results correct? |
| **Recall@K**                | Coverage | Did we retrieve the right history item at all? |
| **NDCG@K**                  | Ranking | How early does the correct result appear? |
| **Reciprocal Rank (MRR)**   | Rank position | On average, how soon is the right result found? |
| **Average Precision (MAP)** | Combined quality | Balances ranking and precision consistency |

---

**Note:**  
Threshold filtering: when a similarity threshold is used (default: distance > 10), only documents above the threshold are considered candidates before sorting and truncating to top-K.
