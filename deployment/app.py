import os
import sys
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import gdown

# ---------------- Add parent directory to path ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------- Import model after sys.path fix ----------------
from models.model import get_efficientnet_b3_model  # EfficientNet B3 function

# ==================== Device ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Google Drive Download ====================
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "1dxqzIyO_xsABDmPSWhGEcfkw2CFOjgzA"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    model = get_efficientnet_b3_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ==================== Transform ====================
custom_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ==================== Streamlit UI ====================
st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩺",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF;'>Chest X-ray Classification</h1>
    <p style='text-align: center; color: gray;'>Predict if an X-ray image shows NORMAL or PNEUMONIA</p>
    """, unsafe_allow_html=True
)

st.write("---")

uploaded_file = st.file_uploader(
    "Upload an X-ray image (PNG, JPG, JPEG)",
    type=["png","jpg","jpeg"],
    accept_multiple_files=False
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing... 🩺"):
            tensor = custom_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                _, pred = torch.max(output, 1)

            classes = ["NORMAL", "PNEUMONIA"]
            result = classes[pred.item()]

            if result == "NORMAL":
                st.success(f"✅ Prediction: {result}")
            else:
                st.error(f"⚠️ Prediction: {result}")

        st.write("---")
        st.info("Note: This model is for research/demo purposes. Consult a medical professional for diagnosis.")
else:
    st.warning("Please upload an X-ray image to get a prediction.")
