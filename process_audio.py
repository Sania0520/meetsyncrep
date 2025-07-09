# process_audio.py


# process_audio.py

import whisper, pymongo, tempfile, os, time, json, uuid
from resemblyzer import VoiceEncoder, preprocess_wav
from pyannote.audio import Pipeline
from pydub import AudioSegment
from datetime import timedelta
import numpy as np

def process_meeting(file_path):
    transcriber = whisper.load_model("base")
    encoder = VoiceEncoder()
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HF_TOKEN"))
    MONGO_URI = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(MONGO_URI)
    client = pymongo.MongoClient("mongodb+srv://Sania:sania%40667@sania.hdopmu9.mongodb.net/?retryWrites=true&w=majority&appName=Sania")
    db = client["MeetSync"]
    collection = db["employee_embeddings"]

    diarization = pipe(file_path)
    audio = AudioSegment.from_wav(file_path)
    output = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
        segment = audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            segment.export(f.name, format="wav")
            wav = preprocess_wav(f.name)
            embed = encoder.embed_utterance(wav)

        best_match, best_sim = None, -1
        for doc in collection.find():
            stored_embed = np.array(doc["embedding"])
            sim = np.dot(embed, stored_embed) / (np.linalg.norm(embed) * np.linalg.norm(stored_embed))
            if sim > best_sim:
                best_sim, best_match = sim, doc

        speaker_name = best_match["name"] if best_sim > 0.75 else f"Unknown ({speaker})"
        result = transcriber.transcribe(f.name, language="en", fp16=False)
        text = result["text"].strip()
        os.remove(f.name)

        output.append(f"[{timedelta(seconds=int(turn.start))} - {timedelta(seconds=int(turn.end))}] {speaker_name}: {text}")

    os.remove(file_path)

    # Save output to temp JSON file
    uid = uuid.uuid4().hex
    json_path = f"/tmp/transcript_{uid}.json"
    with open(json_path, "w") as f:
        json.dump({"transcript": output}, f)

    return uid, output

