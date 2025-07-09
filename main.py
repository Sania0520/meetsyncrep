# main.py


from fastapi import FastAPI, UploadFile, File
from process_audio import process_meeting

app = FastAPI()

@app.post("/process_meeting")
async def process_meeting_endpoint(file: UploadFile = File(...)):
    import tempfile

    # Save file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        file_id, transcript = process_meeting(tmp_path)
        return {"file_id": file_id, "transcript": transcript}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_transcript/{file_id}")
def get_transcript(file_id: str):
    path = f"/tmp/transcript_{file_id}.json"
    import os, json
    if not os.path.exists(path):
        return {"error": "Not found"}
    with open(path) as f:
        return json.load(f)

