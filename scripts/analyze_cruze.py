import fitz
import sys

def analyze(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Total pages: {doc.page_count}")
    
    # Analyze first 50 pages for TOC
    print("--- Searching for Table of Contents (Pages 1-50) ---")
    for i in range(50):
        text = doc[i].get_text()
        if "TABLE OF CONTENTS" in text.upper():
            print(f"Potential TOC on page {i+1}")
            print(text[:500])
            print("-" * 20)
    
    # Check for TOC
    try:
        toc = doc.get_toc()
        print(f"\n--- PDF TOC found: {len(toc) > 0} ---")
        if toc:
            print(f"Total TOC entries: {len(toc)}")
            print("First 50 TOC entries:")
            for entry in toc[:50]:
                print(f"Level {entry[0]}: {entry[1]} (Page {entry[2]})")
    except Exception as e:
        print(f"Error getting TOC: {e}")
    for i in range(0, doc.page_count, 500):
        page = doc[i]
        text = page.get_text()
        lines = text.strip().split('\n')
        if len(lines) >= 3:
            # Footer is usually the last few lines
            print(f"Page {i+1} Footer Area:")
            print("\n".join(lines[-5:]))
            print("-" * 20)
            
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python3 analyze_cruze.py <pdf_path>")
