#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poetry Wrapper: Scrape -> Higgs TTS -> Video Generation
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Imports
from scraper import HorDeScraper
from tts_higgs_backend import HiggsTTSBackend
from video_engine import VideoEngine 

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = BASE_DIR / "outputs"

# --- TEST MODE ---
# Set to True for fast testing with a short poem
USE_TEST_POEM = False 
# -----------------

def main():
    print("[WRAPPER] Starting PoetryBot Pipeline...")

    poem = None

    if USE_TEST_POEM:
        print("[TEST MODE] Using hardcoded short poem.")
        poem = {
            'title': "Der Test",
            'author': "Bot",
            'text': "Dies ist ein Test für das Video.\nEs ist kurz und rendert schnell.\nDie Bilder sollen schön sein.\nUnd der Ton ganz hell.",
            'url': "http://test.local",
            'url_hash': "test_12345"
        }
    else:
        # 1. Scrape (Real Mode)
        scraper = HorDeScraper()
        for _ in range(3):
            poem = scraper.fetch_random_poem()
            if poem: break
        
        if not poem:
            print("[ERROR] Scraping failed.")
            sys.exit(1)

    print(f"[WRAPPER] Poem: '{poem['title']}' by {poem['author']}")

    # 2. TTS Generation
    tts_backend = HiggsTTSBackend()
    
    try:
        audio_path, segments = tts_backend.synthesize_poem(
            poem_data=poem,
            outputs_root=OUTPUTS_ROOT,
        )
    except Exception as e:
        print(f"[ERROR] TTS Failed: {e}")
        sys.exit(1)
        
    print(f"[WRAPPER] Audio ready: {audio_path}")

    # 3. Video Generation
    print("[WRAPPER] Starting Video Generation...")
    video_engine = VideoEngine()
    
    try:
        video_path = video_engine.create_video_from_segments(
            segments=segments, 
            output_filename=f"poem_{poem['url_hash']}",
            full_poem_text=poem['text']  # <--- Das hat gefehlt!
        )
        print(f"[WRAPPER] Video ready: {video_path}")
    except Exception as e:
        print(f"[ERROR] Video Generation Failed: {e}")
        video_path = None

    # 4. JSON Output
    result = {
        "status": "success",
        "title": poem["title"],
        "author": poem["author"],
        "text": poem["text"],
        "url_hash": poem["url_hash"],
        "audio_path": str(audio_path),
        "video_path": str(video_path) if video_path else None,
        "generated_at": datetime.now().isoformat(),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
