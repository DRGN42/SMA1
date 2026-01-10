#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poetry Scraper for hor.de
Fetches random poem from https://hor.de/gedichte/gedicht.php
Optimized to extract correct Author/Title from HTML (with Umlauts).
"""

import requests
from bs4 import BeautifulSoup
import json
import sys
import re
import hashlib
from datetime import datetime
from pathlib import Path

class HorDeScraper: 
    """Scraper for hor.de Poem Archive"""
    
    def __init__(self):
        self.base_url = "https://hor.de/gedichte/gedicht.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
        })
    
    def fetch_random_poem(self):
        """Fetches a random poem from hor.de"""
        try:
            print(f"[INFO] Fetching from {self.base_url}...")
            
            response = self.session.get(
                self.base_url,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            final_url = response.url
            print(f"[INFO] Redirected to: {final_url}")
            
            poem_data = self._parse_html(response.text, final_url)
            
            if poem_data:
                poem_data['url_hash'] = hashlib.sha256(final_url.encode()).hexdigest()[:16]
                poem_data['scraped_at'] = datetime.now().isoformat()
                print(f"[OK] Scraped: '{poem_data['title']}' by {poem_data['author']}")
                return poem_data
            else:
                print("[ERROR] Could not parse poem from HTML")
                return None
                
        except requests.RequestException as e:
            print(f"[ERROR] HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_same_name(self, url_name: str, html_name: str) -> bool:
        """Prüft ob 'Hoelty' und 'Hölty' wahrscheinlich gleich sind"""
        def normalize(s):
            return s.lower().replace('ä','ae').replace('ö','oe').replace('ü','ue').replace('ß','ss').replace(' ','')
        
        n_url = normalize(url_name)
        n_html = normalize(html_name)
        return (n_url in n_html or n_html in n_url) and len(n_html) > 2

    def _parse_html(self, html: str, url: str) -> dict:
        """Parses the poem page from hor.de using smart Name/Title extraction."""
        soup = BeautifulSoup(html, 'lxml')
        
        # 1. Fallback Metadata from URL
        title_url = "Unbekanntes Gedicht"
        author_url = "Unbekannter Autor"
        
        url_match = re.search(r'/gedichte/([^/]+)/([^/]+)\.htm', url)
        if url_match:
            author_url = url_match.group(1).replace('_', ' ').title()
            title_url = url_match.group(2).replace('_', ' ').title()

        # 2. Extract Real Metadata from HTML (to get Umlaute right)
        real_author = author_url
        real_title = title_url
        
        # Search for Author Link
        for a in soup.find_all('a'):
            link_text = a.get_text(strip=True)
            if self._is_same_name(author_url, link_text):
                real_author = link_text 
                break
        
        # Search for Title (bold/h1)
        for b in soup.find_all(['b', 'h1', 'strong']):
            b_text = b.get_text(strip=True)
            if self._is_same_name(title_url, b_text):
                real_title = b_text
                break

        title = real_title
        author = real_author
        
        # 3. Text Extraction (Paragraph Filtering)
        poem_lines = []
        paragraphs = soup.find_all('p')
        
        for p in paragraphs:
            # Skip Navigation
            if p.find('a', href=re.compile(r'(\.\./|index|hor\.de|listen|notizen)')):
                text_clean = p.get_text(strip=True)
                if text_clean.lower() == author.lower():
                    continue
                if len(text_clean) == len("".join([a.get_text(strip=True) for a in p.find_all('a')])):
                     continue

            text_clean = p.get_text(strip=True)
            if not text_clean: continue
            
            # Skip Title/Author in body
            if text_clean.lower() == title.lower() or text_clean.lower() == author.lower():
                continue

            for br in p.find_all('br'): br.replace_with('\n')
            
            strophe_text = p.get_text()
            lines = [l.strip() for l in strophe_text.split('\n') if l.strip()]
            
            filtered_lines = []
            for line in lines:
                if any(x in line.lower() for x in ['impressum', 'kontakt', 'datenschutz', 'hor.de']): continue
                filtered_lines.append(line)

            if filtered_lines:
                poem_lines.extend(filtered_lines)
                poem_lines.append("")

        poem_text = "\n".join(poem_lines).strip()
        
        # FALLBACK Raw Extraction
        if not poem_text or len(poem_text) < 20:
             print("[INFO] Fallback to raw text extraction...")
             body = soup.find('body')
             if body:
                for tag in body.find_all(['script', 'style', 'nav', 'header', 'footer']): tag.decompose()
                raw_text = body.get_text(separator='\n')
                poem_text = self._extract_poem_text_fallback(raw_text, title, author)

        if poem_text: poem_text = self._clean_text(poem_text)
        
        if not poem_text or len(poem_text.strip()) < 20:
            print("[WARNING] Poem text too short or empty")
            return None
        
        return {
            'title': title,
            'author': author,
            'text':  poem_text,
            'url': url
        }
    
    def _extract_poem_text_fallback(self, raw_text: str, title: str, author: str) -> str:
        """Fallback: Extract text from raw body string"""
        lines = raw_text.split('\n')
        poem_lines = []
        skip_patterns = [
            'impressum', 'copyright', 'startseite', 'navigation',
            'home', 'kontakt', 'email', 'index', 'gedichte von',
            'alle rechte', 'http://', 'https://', 'www.', 'hor.de',
            'gedichtsammlung', 'wörterlisten', 'notizen'
        ]
        
        for line in lines:
            line_stripped = line.strip()
            if not poem_lines and not line_stripped: continue
            if line_stripped.lower() == title.lower(): continue
            if line_stripped.lower() == author.lower(): continue
            if any(skip in line_stripped.lower() for skip in skip_patterns): continue
            if len(line_stripped) < 2 and line_stripped not in ['']: continue
            poem_lines.append(line_stripped)
        
        return '\n'.join(poem_lines)
    
    def _clean_text(self, text: str) -> str:
        """Cleans up whitespace"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        while '\n\n\n' in text: text = text.replace('\n\n\n', '\n\n')
        lines = [line.rstrip() for line in text.split('\n')]
        while lines and not lines[0].strip(): lines.pop(0)
        while lines and not lines[-1].strip(): lines.pop()
        return '\n'.join(lines)


def main():
    """Main function"""
    scraper = HorDeScraper()
    poem = scraper.fetch_random_poem()
    
    if poem:
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(json.dumps(poem, indent=2, ensure_ascii=False))
        
        output_file = Path(f'/opt/poetrybot/outputs/logs/poem_{poem["url_hash"]}.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(poem, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Saved to:  {output_file}")
        
        sys.exit(0)
    else:
        print("[FATAL] Failed to fetch poem")
        sys.exit(1)


if __name__ == '__main__':
    main()
