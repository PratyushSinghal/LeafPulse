import streamlit as st
import numpy as np
import base64  
import os  
from openai import OpenAI
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

background_image = 'bg.png'

# Function to calculate the stomata ratio
def calculate_stomata_ratio(results):
    open_stomata_count = 0
    closed_stomata_count = 0

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls) 
            if result.names[class_id] == 'Open_Stomata':
                open_stomata_count += 1
            elif result.names[class_id] == 'Closed_Stomata':
                closed_stomata_count += 1

    total_stomata_count = open_stomata_count + closed_stomata_count

    if total_stomata_count == 0:
        ratio = 0
    else:
        ratio = open_stomata_count / total_stomata_count

    return ratio, open_stomata_count, closed_stomata_count

# Function to classify plant health
def classify_plant_health(ratio):
    if ratio < 0.4:
        return "Overwatered. Reduce water"
    elif 0.4 <= ratio <= 0.8:
        return "Healthy. Optimum irrigation"
    else:
        return "Underwatered. Please water more often"

# Function to make prediction using fine-tuned YOLOv8 model
def make_prediction(image_file):
    model = YOLO('Stomata_Detection_YOLOV8/model/detect/train2/weights/best.pt')
    
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_file.read())
    
    results = model(temp_image_path)
    
    result_image = results[0].plot() 
    
    output_image_path = "output_image_with_boxes.jpg"
    Image.fromarray(result_image).save(output_image_path)
    stomata_ratio, open_stomata_count, closed_stomata_count = calculate_stomata_ratio(results)
    os.remove(temp_image_path)
    plant_health_status = classify_plant_health(stomata_ratio)
    return output_image_path, stomata_ratio, open_stomata_count, closed_stomata_count, plant_health_status

#App
def main():
    st.title("Stomata Detection and Classification")
    if 'plant_health_status' not in st.session_state:
        st.session_state.plant_health_status = "No Data Yet."
    
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url(data:image/jpeg;base64,{base64.b64encode(open(background_image, "rb").read()).decode()});
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        
        # Make prediction on the uploaded image
        if st.button("Make Prediction"):
            output_image_path, stomata_ratio, open_stomata_count, closed_stomata_count, plant_health_status = make_prediction(uploaded_file)

            st.session_state.plant_health_status = plant_health_status
            
            st.image(output_image_path, caption="Detected Objects with Bounding Boxes")
            
            st.write(f"Open Stomata Count: {open_stomata_count}")
            st.write(f"Closed Stomata Count: {closed_stomata_count}")
            st.write(f"Stomata Ratio (Open/Total): {stomata_ratio:.2f}")
            st.write(f"Plant Health Status: {plant_health_status}")
    
    # Chatbot
    st.header("Ask me for advice")
    place = st.text_input("Enter your location:")
    user_input = st.text_area("Type your question or statement here:", "")
    current_month = datetime.now().strftime("%B")
    
    if st.button("Ask"):
        client = OpenAI(api_key= st.secrets["API_key"])
        plant_health_status = st.session_state.get('plant_health_status', 'Unknown')


        prompt = (
            f"You are located in {place}. The plant health status is: {plant_health_status}. "
            f"The current month is {current_month}. Now consider the following question: {user_input}"
        )
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
        )

        
        st.text_area("Input:", user_input)
        st.text_area("Response:", response.choices[0].message['content'].strip())

    # Clear button
    if st.button("Clear"):
        st.text_area("Type your question or statement here:", value="")
        st.text_area("Response:", value="")

if __name__ == "__main__":
    main()

