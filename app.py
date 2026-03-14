from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from groq import Groq
import json
import os

app = Flask(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text(image_file):
    img = Image.open(image_file)
    img = img.convert('L')
    text = pytesseract.image_to_string(img, lang='eng')
    return text.strip()

def structure_medical_data(raw_text):
    prompt = f"""
    You are a medical document parser.
    Extract and return ONLY a JSON object with these fields (use null if not found).

    For a PRESCRIPTION return:
    {{
        "document_type": "prescription",
        "patient_name": "",
        "patient_age": "",
        "patient_gender": "",
        "doctor_name": "",
        "clinic_hospital": "",
        "date": "",
        "diagnosis": "",
        "medicines": [
            {{
                "name": "",
                "dosage": "",
                "frequency": "",
                "duration": "",
                "instructions": ""
            }}
        ],
        "advice": "",
        "follow_up": ""
    }}

    For a LAB REPORT return:
    {{
        "document_type": "lab_report",
        "patient_name": "",
        "patient_age": "",
        "patient_gender": "",
        "doctor_name": "",
        "lab_name": "",
        "date": "",
        "sample_type": "",
        "tests": [
            {{
                "test_name": "",
                "result": "",
                "unit": "",
                "reference_range": "",
                "status": "normal/high/low"
            }}
        ],
        "summary": ""
    }}

    Auto-detect document type. Return ONLY valid JSON, nothing else.

    RAW TEXT:
    {raw_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )

    response_text = response.choices[0].message.content.strip()

    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    return json.loads(response_text.strip())


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "JanArogya AI API is running",
        "version": "1.0",
        "endpoint": "POST /extract — send image file with key 'image'"
    })

@app.route('/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return jsonify({"error": "No image sent. Use key 'image'"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        raw_text = extract_text(file)

        if not raw_text:
            return jsonify({"error": "Could not extract text from image"}), 422

        structured = structure_medical_data(raw_text)
        return jsonify({"success": True, "data": structured})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
