import fitz
import sys

def search_year(pdf_path, year="2014"):
    doc = fitz.open(pdf_path)
    found_pages = []
    for i in range(doc.page_count):
        if year in doc[i].get_text():
            found_pages.append(i + 1)
            if len(found_pages) >= 5:
                break
    doc.close()
    if found_pages:
        print(f"Found {year} on pages: {found_pages}")
    else:
        print(f"{year} not found in the manual.")

if __name__ == "__main__":
    search_year(sys.argv[1])
