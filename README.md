# Adobe India Hackathon - Round 1B  
## 📘 Persona-Based PDF Section Extractor

### 🧠 Problem Statement

Given a set of unstructured PDFs and a user profile (persona and job-to-be-done), the task is to extract and rank the most relevant sections from the documents that would help the user accomplish their job.

---

## 💡 My Approach

### 🔹 1. **Heading Detection**
- Used **PyMuPDF** to extract text blocks and their font sizes.
- Identified headings using:
  - Font size ≥ 12pt
  - Short line (≤ 10 words)
- Grouped following paragraphs under the nearest heading.

### 🔹 2. **Section Extraction**
- Each heading and its content is treated as a separate “section”.
- Preserved the page number and section title for traceability.

### 🔹 3. **Relevance Scoring**
- Combined `persona` + `job_to_be_done` to form a search query.
- Used `TF-IDF` vectorization (from `scikit-learn`) to compute relevance of each section to the query.
- Ranked the top sections using **cosine similarity**.

### 🔹 4. **Output Format**
- Returned the top 3 ranked sections per document.
- Followed the expected `result.json` structure exactly.

---

## 🔧 How to Build and Run

### 🔹 Step 1: Build Docker Image

```bash
docker build --platform linux/amd64 -t roundb-solution .
