import os
import re
import requests
from bs4 import BeautifulSoup
import markdownify
from urllib.parse import urljoin

# Configure
BASE_URL = "https://www.business.hsbc.com.hk/en-gb/products-and-solutions"  # Replace with the website URL to crawl
OUTPUT_DIR = "./data"    # Directory to save Markdown files
VISITED_URLS = set()              # To keep track of visited URLs

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def fetch_page(url):
    """Fetch the HTML content of a web page."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_content(html):
    """Extract relevant content from the HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    return str(main_content) if main_content else ""

def html_to_markdown(html):
    """Convert HTML content to Markdown."""
    return markdownify.markdownify(html, heading_style="ATX")

def sanitize_filename(filename):
    """Sanitize the filename by replacing invalid characters with underscores."""
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = filename.strip(". ")
    return filename

def save_markdown(url, content):
    """Save Markdown content to a file."""
    filename = url.replace(BASE_URL, "").strip("/")
    if not filename:
        filename = "index"  # Default filename for the root URL
    filename = sanitize_filename(filename)
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.md")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Saved: {filepath}")

def extract_links(html, base_url):
    """Extract all internal links from the page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(base_url, href)
        if full_url.startswith(base_url):
            links.add(full_url)
    return links

def crawl_website(url, base_url):
    """Crawl a website and save its content as Markdown files."""
    if url in VISITED_URLS:
        return
    VISITED_URLS.add(url)

    print(f"Crawling: {url}")
    html = fetch_page(url)
    if html:
        content = extract_content(html)
        markdown_content = html_to_markdown(content)
        save_markdown(url, markdown_content)

        internal_links = extract_links(html, base_url)
        for link in internal_links:
            crawl_website(link, base_url)

if __name__ == "__main__":
    crawl_website(BASE_URL, BASE_URL)
