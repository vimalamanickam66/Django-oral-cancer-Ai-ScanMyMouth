"""
ScanMyMouth AI — predict.py
FIXED: No longer duplicates model loading logic.
       Delegates entirely to OralCancerDetector in ai_engine.py.

Standalone usage:
    python predict.py path/to/image.jpg
"""
import os
import sys


def predict_oral_cancer(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    except OSError as e:
        return {"error": f"Cannot read file: {e}"}

    try:
        from oral_detection.ai_engine import OralCancerDetector
        return OralCancerDetector.get_instance().predict(image_bytes)
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'scanmymouth.settings')
    import django
    django.setup()

    result = predict_oral_cancer(sys.argv[1])

    if 'error' in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"Detection  : {result['detection_result'].upper()}")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Confidence : {result['confidence_score']}%")
    print(f"Cancer     : {result['cancer_probability']}%")
    print(f"Normal     : {result['normal_probability']}%")
    print(f"Urgency    : {result['urgency']}")
    print(f"Time       : {result['processing_time_ms']} ms")
    print(f"\n{result['ai_description']}")