import os
import glob
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# --- CONFIGURATION ---
INPUT_DIR = "./tcpos_manual_html"
OUTPUT_DIR = "./tcpos_clean_md"
MIN_CHAR_COUNT = 50  # The Quality Control Threshold

def clean_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. THE SNIPER: Try to grab only the actual content div
    # "Help & Manual" stores the real text inside <div id="idcontent">
    content_div = soup.find('div', id='idcontent')
    if content_div:
        # If we find it, we replace the whole soup with just this div
        soup = content_div 

    # 2. NUKE THE JUNK
    for element in soup(["script", "style", "meta", "noscript", "header", "footer"]):
        element.decompose()

    for tag in soup.find_all(True):
        tag.attrs = {} 

    cleaned_html = str(soup)
    markdown_text = md(cleaned_html, heading_style="ATX", strip=['a']) 

    # 3. Aggressive Empty Space Cleaning
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip() != ""]
    clean_markdown = "\n".join(lines)

    return clean_markdown

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    html_files = glob.glob(f"{INPUT_DIR}/**/*.htm*", recursive=True)
    print(f"Found {len(html_files)} HTML files. Starting pipeline...")

    saved_count = 0
    skipped_count = 0

    for filepath in html_files:
        try:
            md_content = clean_html_file(filepath)
            
            # 4. THE QUALITY CONTROL GATE
            # If the resulting markdown is just a title or practically empty, skip it!
            if len(md_content) < MIN_CHAR_COUNT:
                # print(f"⏩ Skipping {os.path.basename(filepath)} (Too short: {len(md_content)} chars)")
                skipped_count += 1
                continue
            
            base_name = os.path.basename(filepath)
            new_name = os.path.splitext(base_name)[0] + ".md"
            output_path = os.path.join(OUTPUT_DIR, new_name)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            saved_count += 1
                
        except Exception as e:
            print(f"ERROR: Failed to process {filepath}: {e}")

    print("--- PIPELINE REPORT ---")
    print(f"Saved clean files: {saved_count}")
    print(f"Skipped empty/useless files: {skipped_count}")
    print(f"Done. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()