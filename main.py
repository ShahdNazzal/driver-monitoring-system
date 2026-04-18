
import os
import cv2
import shutil
import gdown
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from model_utils import predict_behavior

app = FastAPI()

# ==============================
# إعداد الموديل (Lazy Load)
# ==============================
FILE_ID = "1TKayUv4XzgFN2kg-QbR8nje6zbArlJZ8"
MODEL_PATH = "best.pt"

model = None  # 🔥 مهم


def get_model():
    global model

    if model is None:
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        if not os.path.exists(MODEL_PATH):
            gdown.download(url, MODEL_PATH, quiet=False)

        model = YOLO(MODEL_PATH)

    return model


# ==============================
# helper
# ==============================
def is_video(filename: str):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


# ==============================
# ENDPOINT
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    model_instance = get_model()  # 🔥 تحميل عند الحاجة فقط

    file_path = "temp_input"
    if is_video(file.filename):
        file_path += ".mp4"
    else:
        file_path += ".jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # IMAGE
    if not is_video(file.filename):
        img = cv2.imread(file_path)

        result = predict_behavior(model_instance, img)

        return {"type": "image", "result": result}

    # VIDEO
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frame_interval = int(fps * 30)

    frame_id = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            result = predict_behavior(model_instance, frame)

            results.append({
                "time_sec": int(frame_id / fps),
                "result": result
            })

        frame_id += 1

    cap.release()

    return {
        "type": "video",
        "total_predictions": len(results),
        "results": results
    }


# ==============================
# RUN (Hugging Face)
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)



