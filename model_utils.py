from classes import CLASSES

def predict_behavior(model, img):

    results = model(img)
    result = results[0]

    # no detections
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "class_id": None,
            "behavior": "No detection",
            "confidence": None
        }

    # 🔥 FORCE CLEAN NUMPY/TENSOR → PYTHON TYPES
    confs = result.boxes.conf.cpu().detach().numpy()
    clss = result.boxes.cls.cpu().detach().numpy()

    best_idx = confs.argmax()

    cls_id = int(clss[best_idx])
    conf = float(confs[best_idx])

    behavior = CLASSES.get(cls_id, f"unknown_{cls_id}")

    return {
        "class_id": cls_id,
        "behavior": str(behavior),
        "confidence": conf
    }


