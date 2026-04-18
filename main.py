import os
import cv2
import gdown
import gradio as gr
from ultralytics import YOLO
from model_utils import predict_behavior

# ==============================
# Model Setup (Lazy Load)
# ==============================
FILE_ID = "1TKayUv4XzgFN2kg-QbR8nje6zbArlJZ8"
MODEL_PATH = "best.pt"

model = None


def get_model():
    global model

    if model is None:
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        if not os.path.exists(MODEL_PATH):
            gdown.download(url, MODEL_PATH, quiet=False)

        model = YOLO(MODEL_PATH)

    return model


# ==============================
# IMAGE PREDICTION
# ==============================
def predict_image(image):
    if image is None:
        return {"error": "No image provided"}

    model_instance = get_model()

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return predict_behavior(model_instance, img)


# ==============================
# VIDEO PREDICTION
# ==============================
def predict_video(video):
    if video is None:
        return None, {"error": "No video provided"}

    model_instance = get_model()

    # safe path handling (HF + local + gradio)
    video_path = getattr(video, "name", None) or video
    video_path = str(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frame_interval = int(fps * 30)

    frame_id = 0
    results = []

    while True:
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

    # 👉 IMPORTANT: return video + json
    return video_path, {
        "type": "video",
        "total_predictions": len(results),
        "results": results
    }


# ==============================
# GRADIO UI
# ==============================
with gr.Blocks() as demo:

    gr.Markdown("# 🚗 Driver Monitoring AI System")

    # ================= IMAGE =================
    with gr.Tab("📷 Image"):
        image_input = gr.Image(type="numpy")
        image_output = gr.JSON()
        image_btn = gr.Button("Predict")

        image_btn.click(
            fn=predict_image,
            inputs=image_input,
            outputs=image_output
        )

    # ================= VIDEO =================
    with gr.Tab("🎥 Video"):
        video_input = gr.Video()
        video_output_video = gr.Video()   # 👈 عرض الفيديو
        video_output_json = gr.JSON()     # 👈 النتائج

        video_btn = gr.Button("Analyze Video")

        video_btn.click(
            fn=predict_video,
            inputs=video_input,
            outputs=[video_output_video, video_output_json]
        )


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    demo.launch()