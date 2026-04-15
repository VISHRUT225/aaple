import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os
import subprocess
import tempfile
import time
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from collections import Counter

# --- Configuration & Styling ---
st.set_page_config(page_title="PPE Detection Comparison Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; border: 1px solid #3e4451; }
    h1, h2, h3, h4 { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- Folders Setup ---
WORKSPACE_DIR = Path(os.getcwd()).absolute()
RES_VIDEOS_DIR = WORKSPACE_DIR / "res-videos"
RES_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# --- Class Mapping (Requested Dictionary) ---
# We will use this to OVERRIDE model.names to fix the Ear/Head mismatch
CLASS_MAP = {
    0: "Person",
    1: "Ear",
    2: "Earmuffs",
    3: "Face",
    4: "Face-guard",
    5: "Face-mask-medical",
    6: "Foot",
    7: "Tools",
    8: "Glasses",
    9: "Gloves",
    10: "Helmet",
    11: "Hands",
    12: "Head",
    13: "Medical-suit",
    14: "Shoes",
    15: "Safety-suit",
    16: "Safety-vest"
}

# --- Helpers ---
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    # FORCE OVERRIDE model.names with the user's provided dictionary
    # We set it on model.model.names because model.names is a read-only property in newer versions
    if hasattr(model, 'model'):
        model.model.names = CLASS_MAP
    return model

def find_results_folders(root_dir="."):
    root = Path(root_dir)
    return sorted([f.name for f in root.iterdir() if f.is_dir() and f.name.startswith("results")])

def load_results_csv(folder_path):
    csv_path = folder_path / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    return None

def convert_video_to_web(input_path, output_path):
    """Converts video to H.264 MP4 using FFmpeg for browser compatibility."""
    try:
        command = [
            "ffmpeg", "-y", "-i", str(input_path.absolute()),
            "-vcodec", "libx264", "-acodec", "aac",
            "-pix_fmt", "yuv420p", "-b:v", "2M",
            str(output_path.absolute())
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        st.error(f"FFmpeg conversion failed: {e}")
        return False

# --- Dashboard Logic ---
st.title("🛡️ PPE Detection Comparison Analysis")
st.subheader("Head-to-Head Performance Evaluation & Quantitative Insights")

results_folders = find_results_folders(os.getcwd())

if not results_folders:
    st.error("No 'results****' folders found. Please run the dashboard in the directory containing your YOLO outputs.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_models = st.multiselect("Select Models to Compare", results_folders, default=results_folders)
    
    st.divider()
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    
    st.subheader("Video Comparison Options")
    target_fps_logic = st.checkbox("Optimize Speed (1 FPS Inference)", value=True, help="Inference happens once per second, but output maintains original FPS.")
    
    st.divider()
    st.info("Labels are synchronized across all models to fix Ear/Head mismatch.")

if not selected_models:
    st.warning("Please select models to begin analysis.")
    st.stop()

# Load Model Metadata
all_data = {}
for model_name in selected_models:
    folder_path = Path(model_name)
    weights_path = folder_path / "weights" / "best.pt"
    all_data[model_name] = {
        "df": load_results_csv(folder_path),
        "weights": weights_path,
        "model": load_yolo_model(str(weights_path)) if weights_path.exists() else None
    }

# Tabs
tabs = st.tabs(["📊 Training Summary", "🖼️ Image Analysis", "🎥 Video Analysis"])

# --- Tab 1: Training Summary ---
with tabs[0]:
    st.header("Training Results Comparison (from CSV)")
    
    summary_list = []
    for model_name, data in all_data.items():
        df = data["df"]
        if df is not None:
            # Find row with best mAP50
            best_idx = df['metrics/mAP50(B)'].idxmax()
            best_row = df.loc[best_idx]
            
            summary_list.append({
                "Model": model_name,
                "Epochs": len(df),
                "mAP50": f"{best_row['metrics/mAP50(B)']:.4f}",
                "mAP95": f"{best_row['metrics/mAP50-95(B)']:.4f}",
                "Precision": f"{best_row['metrics/precision(B)']:.4f}",
                "Recall": f"{best_row['metrics/recall(B)']:.4f}",
                "Val Box Loss": f"{best_row['val/box_loss']:.4f}",
                "Val Cls Loss": f"{best_row['val/cls_loss']:.4f}",
                "Best Epoch": int(best_row['epoch'])
            })
    
    if summary_list:
        sdf = pd.DataFrame(summary_list)
        st.dataframe(sdf, width='stretch')
        
        # Visual Benchmarks
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(sdf, x="Model", y="mAP50", color="mAP50", title="mAP50 Benchmark"), width='stretch')
        with c2:
            st.plotly_chart(px.bar(sdf, x="Model", y="Precision", color="Precision", title="Precision Benchmark"), width='stretch')
    else:
        st.error("Could not find results.csv in the selected folders.")

# --- Tab 2: Image Analysis ---
with tabs[1]:
    st.header("Live Image Detection Comparison")
    up_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="img_up")
    
    if up_img:
        img_pil = Image.open(up_img)
        if st.button("Run Multi-Model Image Analysis"):
            analysis_data = []
            cols = st.columns(len(selected_models))
            
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.write(f"**{model_name}**")
                    model = all_data[model_name]["model"]
                    if model:
                        res = model.predict(img_pil, conf=conf_threshold, verbose=False)[0]
                        st.image(res.plot(), width='stretch')
                        
                        # Count detections using synchronised CLASS_MAP
                        for c_id, c_name in CLASS_MAP.items():
                            count = (res.boxes.cls == c_id).sum().item()
                            if count > 0:
                                analysis_data.append({"Model": model_name, "Class": c_name, "Count": int(count)})
            
            if analysis_data:
                st.divider()
                st.header("📈 Detection Summary Analysis")
                adf = pd.DataFrame(analysis_data)
                fig_ana = px.bar(adf, x="Class", y="Count", color="Model", barmode="group", title="Total Detections by Class")
                st.plotly_chart(fig_ana, width='stretch')

# --- Tab 3: Video Analysis ---
with tabs[2]:
    st.header("Live Video Analysis (Corrected Timing & Labels)")
    up_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="vid_up")
    
    if up_vid:
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_in.write(up_vid.read())
        t_in.close()
        
        cap = cv2.VideoCapture(t_in.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        stride = int(fps) if target_fps_logic else 1
        st.info(f"Video FPS: {fps:.2f} | Resolution: {width}x{height} | Inference Stride: {stride} frames")

        if st.button("Start Video Comparison Analysis"):
            video_analysis_results = []
            cols = st.columns(len(selected_models))
            
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.write(f"**{model_name}**")
                    model = all_data[model_name]["model"]
                    if model:
                        with st.spinner(f"Analyzing {model_name}..."):
                            # Setup Video Processing
                            cap = cv2.VideoCapture(t_in.name)
                            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                            # Using 'avc1' or standard mp4v. ffmpeg will fix the codec later.
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out_writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
                            
                            frame_idx = 0
                            last_annotated_frame = None
                            model_counts = Counter()
                            
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                
                                # Inference at interval (e.g., 1 per second)
                                if frame_idx % stride == 0:
                                    res = model.predict(frame, conf=conf_threshold, verbose=False)[0]
                                    last_annotated_frame = res.plot()
                                    # Aggregate all detections in this frame
                                    for c_id in res.boxes.cls.tolist():
                                        class_name = CLASS_MAP.get(int(c_id), f"Unknown_{c_id}")
                                        model_counts[class_name] += 1
                                
                                # Write frame (maintain original duration by repeating annotation)
                                if last_annotated_frame is not None:
                                    out_writer.write(last_annotated_frame)
                                else:
                                    out_writer.write(frame)
                                    
                                frame_idx += 1
                            
                            cap.release()
                            out_writer.release()
                            
                            # Final Save & Re-encode
                            final_path = RES_VIDEOS_DIR.absolute() / f"{model_name.replace(' ', '_')}_{int(time.time())}.mp4"
                            if convert_video_to_web(Path(temp_out), final_path):
                                with open(final_path, "rb") as f:
                                    st.video(f.read())
                                st.success(f"Saved: {final_path.name}")
                            
                            # Store aggregation for final video chart
                            for c_name, count in model_counts.items():
                                video_analysis_results.append({"Model": model_name, "Class": c_name, "Total Detected": count})
            
            if video_analysis_results:
                st.divider()
                st.header("📈 Video Performance: Total Detection Volume")
                vdf = pd.DataFrame(video_analysis_results)
                fig_v = px.bar(vdf, x="Class", y="Total Detected", color="Model", barmode="group", title="Detection Counts Over Whole Video")
                st.plotly_chart(fig_v, width='stretch')
