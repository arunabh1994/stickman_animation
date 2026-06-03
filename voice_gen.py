import os
import time
from pathlib import Path
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
from config import ELEVENLABS_API_KEY, VOICE_IDS, ELEVENLABS_MODEL, AUDIO_DIR

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def generate_voice_for_line(line: dict) -> str:
    """
    Generate a .wav file for a single dialogue line.
    Returns the path to the .wav file.
    Skips generation if the file already exists (caching).
    """
    line_id   = line["id"]
    character = line["character"]
    text      = line["text"]
    emotion   = line["emotion"]

    os.makedirs(AUDIO_DIR, exist_ok=True)
    wav_path = os.path.join(AUDIO_DIR, f"{line_id}.wav")
    print(wav_path)

    if os.path.exists(wav_path):
        return wav_path   # Cache hit — reuse existing file
    else:
        pass

    voice_id = VOICE_IDS.get(character)
    print(voice_id)
    if not voice_id:
        raise ValueError(f"No voice ID configured for character '{character}'")

    # Emotion → voice settings (tweak these to your taste)
    emotion_settings = {
        "neutral":    VoiceSettings(stability=0.55, similarity_boost=0.80, style=0.10),
        "happy":      VoiceSettings(stability=0.40, similarity_boost=0.75, style=0.35),
        "laughing":   VoiceSettings(stability=0.35, similarity_boost=0.70, style=0.45),
        "excited":    VoiceSettings(stability=0.35, similarity_boost=0.75, style=0.40),
        "sad":        VoiceSettings(stability=0.70, similarity_boost=0.80, style=0.05),
        "serious":    VoiceSettings(stability=0.70, similarity_boost=0.85, style=0.05),
        "angry":      VoiceSettings(stability=0.30, similarity_boost=0.80, style=0.50),
        "surprised":  VoiceSettings(stability=0.40, similarity_boost=0.75, style=0.30),
        "sarcastic":  VoiceSettings(stability=0.50, similarity_boost=0.75, style=0.25),
        "thinking":   VoiceSettings(stability=0.65, similarity_boost=0.80, style=0.10),
    }
    settings = emotion_settings.get(emotion, emotion_settings["neutral"])

    # Generate audio (ElevenLabs returns an iterator of bytes)
    mp3_path = wav_path.replace(".wav", ".mp3")
    audio_bytes = b"".join(
        chunk for chunk in client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=ELEVENLABS_MODEL,
            voice_settings=settings,
            output_format="mp3_44100_128",
        )
        if chunk
    )

    # Save MP3 temporarily, convert to WAV (Rhubarb requires WAV)
    with open(mp3_path, "wb") as f:
        f.write(audio_bytes)
    print(mp3_path)
    print(wav_path)
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    os.remove(mp3_path)

    time.sleep(0.3)   # Be polite to the API
    return wav_path


def get_audio_duration(wav_path: str) -> float:
    """Return audio duration in seconds."""
    audio = AudioSegment.from_wav(wav_path)
    return len(audio) / 1000.0


def generate_all_voices(lines: list, verbose: bool = True) -> dict:
    """
    Generate voices for every line.
    Returns {line_id: {"path": str, "duration": float}}
    """
    from tqdm import tqdm
    results = {}
    iterator = tqdm(lines, desc="Generating voices") if verbose else lines

    for line in iterator:
        path     = generate_voice_for_line(line)
        duration = get_audio_duration(path)
        results[line["id"]] = {"path": path, "duration": duration}

    return results
