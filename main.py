import os
import fitz  # PyMuPDF
import json
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    
    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            block_text = ""
            max_font = 0
            for line in b["lines"]:
                for span in line["spans"]:
                    block_text += span["text"].strip() + " "
                    max_font = max(max_font, span["size"])
            block_text = block_text.strip()
            if block_text:
                sections.append({
                    "page": page_number,
                    "font_size": max_font,
                    "text": block_text
                })

    # Group into heading + content
    grouped = []
    current = None
    for sec in sections:
        is_heading = (
            sec["font_size"] >= 13 and
            len(sec["text"].split()) <= 8 and
            sec["text"].isalpha()
        )
        if is_heading:
            if current:
                grouped.append(current)
            current = {
                "section_title": sec["text"],
                "page": sec["page"],
                "content": ""
            }
        elif current:
            current["content"] += " " + sec["text"]

    if current:
        grouped.append(current)

    # Fallback if no headings found
    if not grouped:
        grouped.append({
            "section_title": "Full Document",
            "page": 1,
            "content": " ".join([s["text"] for s in sections])
        })

    return grouped


def rank_sections(sections, query, top_n=3):
    contents = [s["content"] for s in sections]
    vectorizer = TfidfVectorizer().fit([query] + contents)
    vectors = vectorizer.transform([query] + contents)
    
    query_vec = vectors[0]
    section_vecs = vectors[1:]
    scores = cosine_similarity(query_vec, section_vecs).flatten()
    
    ranked = []
    for i, score in enumerate(scores):
        ranked.append({
            "section_title": sections[i]["section_title"],
            "page": sections[i]["page"],
            "content": sections[i]["content"],
            "score": score
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    for idx, r in enumerate(ranked[:top_n]):
        r["importance_rank"] = idx + 1
    
    return ranked[:top_n]

def main():
    input_dir = "input"
    output_dir = "output"
    config_path = os.path.join(input_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    persona = config["persona"]
    job = config["job_to_be_done"]
    query = persona + " " + job

    pdfs = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    metadata = {
        "documents": pdfs,
        "persona": persona,
        "job_to_be_done": job,
        "timestamp": str(datetime.datetime.now())
    }

    relevant_sections = []
    subsection_analysis = []

    for pdf_file in pdfs:
        path = os.path.join(input_dir, pdf_file)
        sections = extract_sections(path)
        ranked = rank_sections(sections, query)

        for r in ranked:
            relevant_sections.append({
                "document": pdf_file,
                "page": r["page"],
                "section_title": r["section_title"],
                "importance_rank": r["importance_rank"]
            })
            subsection_analysis.append({
                "document": pdf_file,
                "page": r["page"],
                "refined_text": r["content"]
            })

    result = {
        "metadata": metadata,
        "relevant_sections": relevant_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
