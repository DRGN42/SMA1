import os
import subprocess
import logging
import sys
import random
from pathlib import Path
from typing import List, Tuple, Dict
from pydub import AudioSegment

# --- CONFIGURATION ---
HIGGS_REPO_PATH = Path("/opt/poetrybot/models/higgs/repo") 
PYTHON_EXECUTABLE = "python3" 

REF_AUDIO_NAME = "storyteller" 

ASSETS_DIR = Path("/opt/poetrybot/assets")
AMBIENT_VOLUME_DB = -15 

TEMPERATURE = "0.3"
# ---------------------

class HiggsTTSBackend:
    def __init__(self):
        self.logger = logging.getLogger("HiggsBackend")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[HIGGS-SUB] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.generation_script = HIGGS_REPO_PATH / "examples" / "generation.py"
        if not self.generation_script.exists():
            self.logger.warning(f"WARNING: Script not found: {self.generation_script}")
        
        self._ensure_voice_prompt_is_wav(REF_AUDIO_NAME)

    def _ensure_voice_prompt_is_wav(self, voice_name: str):
        prompts_dir = HIGGS_REPO_PATH / "examples" / "voice_prompts"
        wav_path = prompts_dir / f"{voice_name}.wav"
        mp3_path = prompts_dir / f"{voice_name}.mp3"

        if wav_path.exists(): return
        if mp3_path.exists():
            self.logger.info(f"Converting Voice-Prompt MP3 -> WAV...")
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
            except Exception as e:
                self.logger.error(f"Error MP3 conversion: {e}")

    def _chunk_text(self, text: str, lines_per_chunk: int = 2) -> List[str]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        chunks = []
        for i in range(0, len(lines), lines_per_chunk):
            chunk = " ".join(lines[i:i + lines_per_chunk])
            chunks.append(chunk)
        return chunks

    def _get_random_ambient_track(self) -> Path:
        if not ASSETS_DIR.exists(): return None
        candidates = list(ASSETS_DIR.glob("ambient*.mp3"))
        if not candidates: return None
        return random.choice(candidates)

    def _add_background_music(self, voice_audio: AudioSegment) -> AudioSegment:
        music_path = self._get_random_ambient_track()
        if not music_path: return voice_audio

        try:
            ambient = AudioSegment.from_file(music_path) + AMBIENT_VOLUME_DB
            if len(ambient) < len(voice_audio):
                loops = (len(voice_audio) // len(ambient)) + 1
                ambient = ambient * loops
            ambient = ambient[:len(voice_audio)].fade_out(3000)
            return voice_audio.overlay(ambient, position=0)
        except Exception:
            return voice_audio

    def synthesize_poem(self, poem_data: dict, outputs_root: Path) -> Tuple[Path, List[Dict]]:
        """
        Returns: (final_mixed_wav_path, list_of_segments_for_video)
        """
        poem_text = poem_data['text']
        title = poem_data['title']
        author = poem_data['author']
        url_hash = poem_data['url_hash']
        
        session_id = f"poem_{url_hash}"
        temp_dir = outputs_root / "temp" / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        final_output_dir = outputs_root / "audio"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / f"{session_id}.wav"

        # 1. Define Segments
        intro_text = f"Ein Gedicht von {author}. Titel: {title}."
        outro_text = f"Das war ein Gedicht von {author} mit dem Titel {title}. Folge fuer mehr Gedichte!"
        
        body_chunks = self._chunk_text(poem_text, lines_per_chunk=2)
        
        # Structure: list of (filename_suffix, text_content, type)
        plan = [("intro", intro_text, "intro")]
        for i, chunk in enumerate(body_chunks):
            plan.append((f"chunk_{i:03d}", chunk, "chunk"))
        plan.append(("outro", outro_text, "outro"))

        self.logger.info(f"Processing {len(plan)} segments...")

        segments_for_video = [] # This will store the result list
        generated_wavs = []     # This stores paths for merging

        # 2. Generation Loop
        for name, text, seg_type in plan:
            seg_filename = f"{name}.wav"
            seg_path = temp_dir / seg_filename
            
            # Generate if missing
            if not (seg_path.exists() and seg_path.stat().st_size > 1000):
                self.logger.info(f"Generating {name}...")
                cmd = [
                    PYTHON_EXECUTABLE,
                    str(self.generation_script),
                    "--transcript", text,
                    "--ref_audio", REF_AUDIO_NAME,
                    "--temperature", TEMPERATURE,
                    "--out_path", str(seg_path)
                ]
                try:
                    subprocess.run(cmd, cwd=HIGGS_REPO_PATH, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error {name}: {e.stderr}")
            
            if seg_path.exists() and seg_path.stat().st_size > 1000:
                # Add to video list
                segments_for_video.append({
                    'type': seg_type,
                    'text': text,
                    'audio': seg_path
                })
                # Add to merge list
                generated_wavs.append((seg_type, seg_path))
            else:
                self.logger.warning(f"Segment {name} failed, skipping.")

        if not generated_wavs:
            raise RuntimeError("No audio generated.")

        # 3. Merging Audio (for the podcast version / checking)
        self.logger.info("Merging audio for preview...")
        full_voice = AudioSegment.empty()
        
        short_pause = AudioSegment.silent(duration=600)
        long_pause = AudioSegment.silent(duration=1200)

        for i, (seg_type, wav_path) in enumerate(generated_wavs):
            try:
                seg = AudioSegment.from_wav(wav_path)
                full_voice += seg
                
                # Logic for pauses in the merged file
                # Note: The video engine will do its own concatenation, 
                # but we need this file for music mixing reference length or standalone audio.
                if i < len(generated_wavs) - 1:
                    next_type = generated_wavs[i+1][0]
                    if seg_type == "intro" or next_type == "outro":
                        full_voice += long_pause
                    else:
                        full_voice += short_pause
            except: pass

        # 4. Mix Music
        final_mix = self._add_background_music(full_voice)
        final_mix.export(final_output_path, format="wav")
        self.logger.info(f"DONE Audio: {final_output_path}")

        return final_output_path, segments_for_video
