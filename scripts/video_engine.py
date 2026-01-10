#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
import PIL.Image

# --- MONKEY PATCH FOR MOVIEPY ---
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# --------------------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import whisper 
from openai import OpenAI 

from moviepy.editor import (
    AudioFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
)

# --- CONFIGURATION ---
OUTPUT_VIDEO_DIR = Path("/opt/poetrybot/outputs/video")
TEMP_IMG_DIR = OUTPUT_VIDEO_DIR / "temp_images"

VIDEO_SIZE = (1080, 1920) 

STYLE_SUFFIX = (
    ", dark moody atmosphere, oil painting style, visible brushstrokes, "
    "cinematic dramatic lighting, melancholic, deep colors, masterpiece, "
    "highly detailed, 8k resolution, artstation trends"
)
NEGATIVE_PROMPT = "text, watermark, logo, ugly, deformed, blurry, bad anatomy, cartoon, anime, bright colors, happy"

MODEL_ID = "Lykon/dreamshaper-8" 
WHISPER_MODEL_SIZE = "small" 

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" 
FONT_SIZE = 85               
FONT_COLOR = 'yellow'        
FONT_STROKE_COLOR = 'black'
FONT_STROKE_WIDTH = 6

class VideoEngine:
    def __init__(self):
        self.logger = logging.getLogger("VideoEngine")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[VIDEO] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_IMG_DIR.mkdir(parents=True, exist_ok=True)

        self.pipe = None 
        self.whisper_model = None

    def load_sd_model(self):
        if self.pipe is not None: return
        self.logger.info(f"Loading SD Model: {MODEL_ID} ...")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_attention_slicing()
            self.logger.info("SD Model loaded!")
        except Exception as e:
            self.logger.error(f"Failed to load SD Model: {e}")
            raise e

    def load_whisper_model(self):
        if self.whisper_model is not None: return
        self.logger.info(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")

    def optimize_prompt(self, text: str, type: str, full_poem_context: str = "") -> str:
        """
        Uses LM Studio with FULL CONTEXT awareness to generate prompts.
        """
        if type == 'intro':
            return "close up of an antique weathered poetry book cover, title written in gold, cinematic lighting, magical dust, library background"
        if type == 'outro':
            return "a candle burning down in a dark room, closed book, smoke, melancholic ending, fading light"

        try:
            client = OpenAI(base_url="http://192.168.178.54:1234/v1", api_key="lm-studio")
            
            system_instruction = (
                "You are an AI Art Director. Your task is to visualize a specific part of a German poem. "
                "I will provide the FULL POEM for context, and the SPECIFIC LINES to visualize now. "
                "1. Read the full poem to understand the overall mood, theme, and setting. "
                "2. Create a visual image prompt for the specific lines that fits perfectly into this overall mood. "
                "3. Ensure the artistic style remains consistent (dark, moody, oil painting). "
                "4. Output ONLY the English image prompt. No explanation."
            )
            
            user_content = (
                f"--- FULL POEM CONTEXT ---\n{full_poem_context}\n"
                f"--- END CONTEXT ---\n\n"
                f"TASK: Create an image prompt for these specific lines:\n'{text}'"
            )

            self.logger.info(f"Asking LLM (Context-Aware)...")
            completion = client.chat.completions.create(
                model="gpt-oss-20b", 
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_prompt = completion.choices[0].message.content.strip()
            if ":" in ai_prompt: ai_prompt = ai_prompt.split(":")[-1].strip()
            
            self.logger.info(f"LLM Prompt: {ai_prompt[:50]}...")
            
            return f"{ai_prompt}{STYLE_SUFFIX}"

        except Exception as e:
            self.logger.error(f"LM Studio Error: {e}. Using fallback.")
            return f"visualize the concept of: {text}{STYLE_SUFFIX}"

    def get_word_timestamps(self, audio_path: Path) -> List[Dict]:
        self.load_whisper_model()
        result = self.whisper_model.transcribe(str(audio_path), word_timestamps=True, language="de")
        words = []
        for segment in result['segments']:
            for word_info in segment['words']:
                words.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        return words

    def generate_image(self, prompt_text: str, filename_seed: str) -> Path:
        self.load_sd_model()
        self.logger.info(f"Gen Image: '{prompt_text[:20]}...'")
        generator = torch.Generator("cuda").manual_seed(random.randint(0, 999999))
        image = self.pipe(
            prompt=prompt_text, 
            negative_prompt=NEGATIVE_PROMPT,
            width=544, height=960,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        out_path = TEMP_IMG_DIR / f"{filename_seed}.png"
        image.save(out_path)
        return out_path

    def create_ken_burns_clip(self, image_path: Path, duration: float) -> ImageClip:
        pil_img = Image.open(image_path)
        pil_img = pil_img.resize(VIDEO_SIZE, Image.Resampling.LANCZOS)
        clip = ImageClip(np.array(pil_img)).set_duration(duration)
        zoom_factor = 1.10
        def resize_func(t): return 1 + (zoom_factor - 1) * t / duration
        zoomed_clip = clip.resize(resize_func)
        final_clip = zoomed_clip.crop(
            x_center=zoomed_clip.w / 2, y_center=zoomed_clip.h / 2, 
            width=VIDEO_SIZE[0], height=VIDEO_SIZE[1]
        )
        return final_clip

    def create_synced_subtitles(self, audio_path: Path, total_duration: float) -> List[TextClip]:
        try:
            word_data = self.get_word_timestamps(audio_path)
            clips = []
            for w in word_data:
                word_text = w['word']
                start = w['start']
                end = w['end']
                if start > total_duration: continue
                duration = end - start
                if duration < 0.1: duration = 0.1
                txt_clip = TextClip(
                    word_text, font=FONT_PATH, fontsize=FONT_SIZE, 
                    color=FONT_COLOR, stroke_color=FONT_STROKE_COLOR, stroke_width=FONT_STROKE_WIDTH,
                    method='label', align='center'
                )
                txt_clip = txt_clip.set_position(('center', 0.80), relative=True)
                txt_clip = txt_clip.set_start(start).set_duration(duration)
                clips.append(txt_clip)
            return clips
        except Exception as e:
            self.logger.error(f"Sync Subtitle Error: {e}")
            return []

    def create_video_from_segments(self, segments: List[Dict], output_filename: str, full_poem_text: str) -> Path:
        """
        NEW: Accepts full_poem_text for context awareness
        """
        self.logger.info(f"Rendering Video with {len(segments)} clips...")
        final_clips = []
        
        for i, seg in enumerate(segments):
            raw_text = seg['text']
            audio_path = seg['audio']
            
            # 1. OPTIMIZE PROMPT (Context Aware)
            prompt = self.optimize_prompt(raw_text, seg['type'], full_poem_text)
            
            # 2. GENERATE IMAGE
            img_path = self.generate_image(prompt, f"seg_{i}_{seg['type']}")
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            
            # 3. VIDEO CLIP
            video_clip = self.create_ken_burns_clip(img_path, duration)
            video_clip = video_clip.set_audio(audio_clip)
            
            # 4. SYNC SUBTITLES
            if raw_text.strip():
                word_clips = self.create_synced_subtitles(audio_path, duration)
                if word_clips:
                    video_clip = CompositeVideoClip([video_clip] + word_clips)
            
            final_clips.append(video_clip)
        
        self.logger.info("Concatenating clips...")
        full_video = concatenate_videoclips(final_clips, method="compose")
        out_path = OUTPUT_VIDEO_DIR / f"{output_filename}.mp4"
        
        self.logger.info("Writing video file...")
        full_video.write_videofile(
            str(out_path), fps=24, codec='libx264', audio_codec='aac',
            threads=4, preset='medium'
        )
        return out_path
