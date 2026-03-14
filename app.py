import streamlit as st
from PIL import Image
import pytesseract
from groq import Groq
import json
import os

# Page config
st.set_page_config(
    page_title="JanArogya AI",
    page_icon="🏥",
    layout="centered"
)

# Header
st.title("🏥 JanArogya AI")
st.subheader("Medical Report & Prescription Extractor")
st.markdown("Upload a photo of any **prescription or lab report** to extract structured data instantly.")

# Groq setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def extract_text(image):
    img = image.convert('L')
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

    # return json.loads(response_text.strip())
    try:
        return json.loads(response_text.strip())
    except:
        st.error("AI returned invalid JSON. Raw response below:")
        st.code(response_text)
        raise


# Upload section
uploaded_file = st.file_uploader(
    "Upload report or prescription image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded document", use_column_width=True)

    # Extract button
    if st.button("Extract Medical Data", type="primary"):
        with st.spinner("Reading document..."):

            # Step 1: OCR
            raw_text = extract_text(image)

            if not raw_text:
                st.error("Could not extract text from this image. Try a clearer photo.")
            else:
                # Step 2: Structure with Groq
                with st.spinner("Structuring data with AI..."):
                    try:
                        data = structure_medical_data(raw_text)

                        st.success("Extraction complete!")
                        st.markdown("---")

                        # Show document type badge
                        doc_type = data.get("document_type", "unknown")
                        if doc_type == "prescription":
                            st.markdown("### 📋 Prescription")
                        else:
                            st.markdown("### 🧪 Lab Report")

                        # Patient info
                        st.markdown("#### Patient Information")
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric("Name", data.get("patient_name") or "N/A")
                        col2.metric("Age", data.get("patient_age") or "N/A")
                        col3.metric("Gender", data.get("patient_gender") or "N/A")

                        # Doctor & hospital
                        st.markdown("#### Doctor & Hospital")
                        col4, col5 = st.columns(2)
                        col4.metric("Doctor", data.get("doctor_name") or "N/A")
                        hospital_name = data.get("clinic_hospital") or data.get("lab_name")
                        col5.metric("Hospital / Lab", hospital_name or "N/A")
                        
                        # col4.metric("Doctor", data.get("doctor_name") or "N/A")
                        # col5.metric("Hospital", data.get("clinic_hospital") or "N/A")

                        # Diagnosis
                        if data.get("diagnosis"):
                            st.markdown("#### Diagnosis")
                            st.info(data.get("diagnosis"))

                        # Medicines (prescription)
                        if doc_type == "prescription" and data.get("medicines"):
                            st.markdown("#### Medicines")
                            for med in data["medicines"]:
                                with st.expander(f"💊 {med.get('name', 'Medicine')}"):
                                    col6, col7 = st.columns(2)
                                    col6.write(f"**Dosage:** {med.get('dosage') or 'N/A'}")
                                    col6.write(f"**Frequency:** {med.get('frequency') or 'N/A'}")
                                    col7.write(f"**Duration:** {med.get('duration') or 'N/A'}")
                                    col7.write(f"**Instructions:** {med.get('instructions') or 'N/A'}")

                        # Tests (lab report)
                        if doc_type == "lab_report" and data.get("tests"):
                            st.markdown("#### Test Results")
                            for test in data["tests"]:
                                status = test.get("status", "").lower()
                                if status == "high":
                                    icon = "🔴"
                                elif status == "low":
                                    icon = "🔵"
                                else:
                                    icon = "🟢"
                                with st.expander(f"{icon} {test.get('test_name', 'Test')}"):
                                    col8, col9 = st.columns(2)
                                    col8.write(f"**Result:** {test.get('result') or 'N/A'} {test.get('unit') or ''}")
                                    col9.write(f"**Reference:** {test.get('reference_range') or 'N/A'}")

                        # Advice
                        if data.get("advice"):
                            st.markdown("#### Advice")
                            st.success(data.get("advice"))

                        # Follow up
                        if data.get("follow_up"):
                            st.markdown("#### Follow Up")
                            st.warning(data.get("follow_up"))

                        # Raw JSON download
                        st.markdown("---")
                        st.markdown("#### Raw JSON Output")
                        st.json(data)

                        # Download button
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(data, indent=2, ensure_ascii=False),
                            file_name="medical_output.json",
                            mime="application/json"
                        )

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

