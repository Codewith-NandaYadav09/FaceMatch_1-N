import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import os
try:
    import cv2
except Exception:
    cv2 = None

API_SEARCH = st.sidebar.text_input("API search URL", value="http://localhost:8000/api/search")
API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.title("FaceMatch 1:N — Demo")
st.write("Upload a face image to search the indexed dataset (local FAISS demo).")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
k = st.sidebar.slider("Top k", 1, 20, 5)

if uploaded is not None:
    try:
        img_bytes = uploaded.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="Query image", use_column_width=True)

        if st.button("Search"):
            with st.spinner("Searching..."):
                files = {"file": (uploaded.name, img_bytes, uploaded.type)}
                params = {"k": k}
                resp = requests.post(API_SEARCH, files=files, params=params, timeout=30)
                if resp.status_code != 200:
                    st.error(f"Search failed: {resp.status_code} {resp.text}")
                else:
                    data = resp.json()
                    results = data.get("results", [])
                    if not results:
                        st.info("No results found in index")
                    else:
                        st.subheader(f"Results ({len(results)} matches)")
                        # Display results in a grid
                        cols = st.columns(min(3, len(results)))
                        for idx, r in enumerate(results):
                            with cols[idx % len(cols)]:
                                image_url = r.get("image_url")
                                filename = r.get("filename", r.get("id"))
                                score = r.get("score", 0)
                                
                                # Debug info (collapsible)
                                with st.expander(f"Debug: {filename}"):
                                    st.write(f"ID: {r.get('id')}")
                                    st.write(f"Filename: {filename}")
                                    st.write(f"Image URL: {image_url}")
                                    st.write(f"Score: {score:.4f}")
                                
                                # Prefer local file path if provided
                                file_path = r.get("file_path")
                                displayed = False
                                if file_path:
                                    # try OpenCV first for faster read
                                    try:
                                        if cv2 is not None and os.path.exists(file_path):
                                            bgr = cv2.imread(file_path)
                                            if bgr is not None:
                                                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                                matched_img = Image.fromarray(rgb)
                                                st.image(matched_img, caption=f"**{filename}**\nScore: {score:.4f}", use_column_width=True)
                                                displayed = True
                                        # fallback to PIL
                                        if not displayed and os.path.exists(file_path):
                                            matched_img = Image.open(file_path)
                                            st.image(matched_img, caption=f"**{filename}**\nScore: {score:.4f}", use_column_width=True)
                                            displayed = True
                                    except Exception as e:
                                        st.warning(f"**{filename}** — Score: {score:.4f}\n(Error loading local file: {str(e)})")

                                # If not displayed via local path, try HTTP image_url
                                if not displayed and image_url:
                                    full_url = f"{API_BASE}{image_url}"
                                    try:
                                        img_resp = requests.get(full_url, timeout=5)
                                        if img_resp.status_code == 200:
                                            matched_img = Image.open(BytesIO(img_resp.content))
                                            st.image(matched_img, caption=f"**{filename}**\nScore: {score:.4f}", use_column_width=True)
                                            displayed = True
                                        else:
                                            st.warning(f"**{filename}** — Score: {score:.4f}\n(HTTP {img_resp.status_code}: image not found at {full_url})")
                                    except Exception as e:
                                        st.warning(f"**{filename}** — Score: {score:.4f}\n(Error fetching image over HTTP: {str(e)})")

                                if not displayed:
                                    st.warning(f"**{filename}** — Score: {score:.4f}\n(No image available)")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.write("Make sure your FastAPI server is running at the API URLs above.")
st.sidebar.write("**Troubleshooting:** Check debug info (expander) for each result to see image URLs and errors.")
