import fitz
import sys

def search_tables(pdf_path, max_pages=500):
    doc = fitz.open(pdf_path)
    for i in range(max_pages):
        text = doc[i].get_text()
        if "Table " in text:
            print(f"Table found on page {i+1}")
            # print surrounding lines
            lines = text.split('\n')
            for line in lines:
                if "Table " in line:
                    print(f"Line: {line}")
            print("-" * 20)
            if i > 100: # find a few
                 break
    doc.close()

if __name__ == "__main__":
    search_tables(sys.argv[1])
