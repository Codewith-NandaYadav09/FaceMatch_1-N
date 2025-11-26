import streamlit as st
import requests
from io import BytesIO
from PIL import Image

API_SEARCH = st.sidebar.text_input("API search URL", value="http://localhost:8000/api/search")

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
                        st.info("No results")
                    else:
                        st.subheader("Results")
                        for r in results:
                            st.write(f"**id:** {r.get('id')} — **score:** {r.get('score'):.4f}")
                            meta = r.get("metadata") or {}
                            if meta:
                                st.write(meta)
    except Exception as e:
        st.error(f"Error processing image: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Make sure your FastAPI server is running at the API URL above.")
