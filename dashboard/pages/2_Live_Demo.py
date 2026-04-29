import streamlit as st
import os
import sys
import time

try:
    import cv2
    import numpy as np
    from pathlib import Path
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

if not _has_cv2:
    st.title("🛣️ Highway Safety AI — Live Demo")
    st.warning(
        "**Live Demo requires OpenCV — available locally only.**\n\n"
        "Run locally:\n"
        "```\ngit clone https://github.com/jay-ogene/highway-safety-ai\n"
        "conda activate highway-safety\n"
        "streamlit run dashboard/app.py\n```\n\n"
        "**Or use the live API:**\n"
        "https://highway-safety-api-720304622514.us-central1.run.app/docs"
    )
    st.stop()

# rest of file continues...

st.title("🛣️ Highway Safety AI — Live Demo")
st.caption("Upload any highway footage and get a real-time incident analysis report")

# ── Sidebar info ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-jam.png", width=60)
    st.title("How it works")
    st.markdown("""
    1. **Upload** a highway video
    2. **YOLOv8s** detects vehicles in each frame
    3. **ByteTrack** assigns persistent IDs
    4. **TTC analysis** flags near-misses
    5. **Behavior classifier** detects dangerous driving
    6. Download annotated video + report
    """)
    st.divider()
    st.caption("Model: YOLOv8s — 0.833 mAP@0.5")
    st.caption("Trained on 140,000 highway frames")
    st.caption("Validated: 8.3M NGSIM observations")
    st.divider()
    confidence = st.slider("Detection confidence", 0.2, 0.8, 0.4, 0.05)
    #max_frames = st.slider("Max frames to process", 50, 300, 150, 25)
    max_frames = st.slider(
    "Max frames to process", 
    50, 500, 200, 25,
    help="Higher = more complete analysis but slower. "
         "200 frames ≈ 8 seconds of footage at 25fps."
    )
    frames_warning = max_frames
    total_duration = max_frames / 25
    st.caption(
        f"Will analyse {max_frames} frames "
        f"= {total_duration:.0f}s of footage. "
        f"Processing time: ~{max_frames * 0.18:.0f}s on CPU."
    )
    #st.caption("More frames = slower but more complete analysis")


# ── Alert level explainer ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.markdown("🟢 **SAFE** — TTC > 3.0s")
col2.markdown("🔵 **WARNING** — TTC 2.5-3.0s")
col3.markdown("🟠 **HIGH RISK** — TTC 1.5-2.5s")
col4.markdown("🔴 **CRITICAL** — TTC < 1.5s")
st.divider()


# ── Upload section ────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload highway footage",
    type=["mp4", "avi", "mov", "mkv"],
    help="Best results with overhead or elevated camera angle. "
         "Dashcam footage also works."
)

if uploaded is None:
    st.info(
        "👆 Upload a video above to begin. "
        "For best results use overhead highway camera footage "
        "similar to traffic CCTV. Dashcam footage also works "
        "but speed estimates will be less accurate."
    )

    # Show sample UA-DETRAC result as a teaser
    st.subheader("Example output")
    st.markdown("""
    When you upload a video, you will receive:
    - **Annotated video** with bounding boxes, vehicle IDs, and speed labels
    - **Incident report** listing every near-miss, tailgating event, 
      sudden brake, and wrong-way vehicle detected
    - **Summary statistics** including minimum TTC, total incidents, 
      and vehicles tracked
    """)

    sample_stats = {
        "Vehicles tracked": "7-17 simultaneously",
        "Near-miss events": "297 CRITICAL in 150 frames",
        "Min TTC recorded": "0.13 seconds",
        "Behaviors detected": "Tailgating · Braking · Wrong-way",
    }
    cols = st.columns(4)
    for i, (k, v) in enumerate(sample_stats.items()):
        cols[i].metric(k, v)

    st.stop()


# ── Process uploaded video ────────────────────────────────────────
if uploaded:
    # Save upload to temp file
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    # Show video info
    cap = cv2.VideoCapture(input_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_original = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_s = total_video_frames / fps_original if fps_original > 0 else 0

    st.success(f"Video uploaded: **{uploaded.name}**")
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Resolution", f"{width}×{height}")
    info2.metric("Total frames", f"{total_video_frames:,}")
    info3.metric("Frame rate", f"{fps_original:.0f} fps")
    info4.metric("Duration", f"{duration_s:.1f}s")

    frames_to_process = min(max_frames, total_video_frames)
    st.caption(
        f"Will process {frames_to_process} of {total_video_frames} frames "
        f"({frames_to_process/fps_original:.1f}s of footage)"
    )

    if st.button("🚀 Run Incident Analysis", type="primary"):

        output_dir = tempfile.mkdtemp()
        progress_bar = st.progress(0, text="Initialising pipeline...")
        status = st.empty()
        start_time = time.time()

        try:
            # Load model
            status.info("Loading YOLOv8s model...")
            from ultralytics import YOLO
            from detection.tracker import VehicleTracker
            from detection.near_miss import NearMissDetector, AlertLevel
            from detection.behavior_classifier import BehaviorClassifier

            model = YOLO("models/best_yolov8s_v1.pt")
            tracker   = VehicleTracker(fps=fps_original, pixels_per_meter=8.0)
            nm_det    = NearMissDetector(pixels_per_meter=8.0, fps=fps_original)
            classifier = BehaviorClassifier(fps=fps_original, pixels_per_meter=8.0)

            # Output video writer
            out_path = os.path.join(output_dir, "analysis_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                out_path, fourcc, fps_original, (width, height)
            )

            cap = cv2.VideoCapture(input_path)
            all_nm_events  = []
            all_beh_events = []
            frame_idx = 0

            COLOR_BOX = {
                AlertLevel.CRITICAL: (0, 0, 255),
                AlertLevel.HIGH:     (0, 100, 255),
                AlertLevel.WARNING:  (0, 165, 255),
                AlertLevel.SAFE:     (200, 200, 200),
            }

            status.info(f"Processing {frames_to_process} frames...")

            while cap.isOpened() and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO detection
                results = model(
                    frame, conf=confidence, verbose=False
                )[0]

                dets = []
                confs = []
                if results.boxes is not None and len(results.boxes):
                    for box, c in zip(
                        results.boxes.xyxy.cpu().numpy(),
                        results.boxes.conf.cpu().numpy()
                    ):
                        dets.append(box)
                        confs.append(c)

                dets  = np.array(dets)  if dets  else np.empty((0,4))
                confs = np.array(confs) if confs else np.empty((0,))

                # Track
                tracks = tracker.update(dets, confs, frame_idx)

                # Near-miss
                nm_events = nm_det.update(tracks, frame_idx)
                all_nm_events.extend(nm_events)

                # Behavior
                beh_events = classifier.update(tracks, frame_idx)
                for i in range(len(tracks)):
                    for j in range(i+1, len(tracks)):
                        e = classifier.check_tailgating(
                            tracks[i], tracks[j], frame_idx
                        )
                        if e:
                            beh_events.append(e)
                all_beh_events.extend(beh_events)

                # Draw boxes and labels
                alert_map = {}
                for e in nm_events:
                    alert_map[e.vehicle_a_id] = e.alert_level
                    alert_map[e.vehicle_b_id] = e.alert_level

                for track in tracks:
                    x1,y1,x2,y2 = track.bbox.astype(int)
                    alert = alert_map.get(track.track_id, AlertLevel.SAFE)
                    color = COLOR_BOX[alert]
                    thick = 3 if alert == AlertLevel.CRITICAL else \
                            2 if alert == AlertLevel.HIGH else 1
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)
                    label = f"ID:{track.track_id} {track.speed_kmh:.0f}km/h"
                    lw, lh = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                    )[0]
                    ly = max(y1-4, lh+4)
                    cv2.rectangle(frame,
                                 (x1, ly-lh-4), (x1+lw+4, ly+2),
                                 color, -1)
                    cv2.putText(frame, label, (x1+2, ly-2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                               (255,255,255), 1, cv2.LINE_AA)

                # Draw alerts
                active = nm_events[:2] + beh_events[:1]
                for i, ev in enumerate(active):
                    y = 22 + i * 24
                    cv2.rectangle(frame, (0, y-16),
                                 (frame.shape[1], y+6), (0,0,0), -1)
                    desc = getattr(ev, 'description', str(ev))
                    cv2.putText(frame, desc[:85], (8, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                               (0,0,255) if 'NEAR' in desc.upper()
                               else (0,165,255),
                               1, cv2.LINE_AA)

                # Frame counter
                cv2.putText(frame,
                           f"Frame {frame_idx+1}/{frames_to_process} "
                           f"| Vehicles: {len(tracks)}",
                           (8, frame.shape[0]-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                           (100,255,100), 1, cv2.LINE_AA)

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 10 == 0:
                    pct = frame_idx / frames_to_process
                    elapsed = time.time() - start_time
                    eta = (elapsed / pct) * (1-pct) if pct > 0 else 0
                    progress_bar.progress(
                        pct,
                        text=f"Processing frame {frame_idx}/"
                             f"{frames_to_process} "
                             f"— ETA: {eta:.0f}s"
                    )

            cap.release()
            writer.release()
            elapsed_total = time.time() - start_time

            progress_bar.progress(1.0, text="Analysis complete!")
            status.success(
                f"Processed {frame_idx} frames in "
                f"{elapsed_total:.1f} seconds"
            )

            st.divider()

            # ── RESULTS ──────────────────────────────────────────
            st.subheader("Incident Report")

            nm_sum  = nm_det.get_summary()
            beh_sum = classifier.get_summary()

            # KPI row
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Frames processed", f"{frame_idx}")
            k2.metric("🔴 Critical",
                      nm_sum.get('by_level',{}).get('CRITICAL',0))
            k3.metric("🟠 High risk",
                      nm_sum.get('by_level',{}).get('HIGH',0))
            k4.metric("Min TTC",
                      f"{nm_sum.get('min_ttc',0):.2f}s"
                      if nm_sum.get('min_ttc') else "none")
            k5.metric("Avg TTC",
                      f"{nm_sum.get('avg_ttc',0):.2f}s"
                      if nm_sum.get('avg_ttc') else "none")
            k6.metric("Behaviors",
                      beh_sum.get('total_behaviors',0))

            # Behavior breakdown
            if beh_sum.get('by_type'):
                st.subheader("Behaviors detected")
                bc = st.columns(len(beh_sum['by_type']))
                for i, (k,v) in enumerate(
                    beh_sum['by_type'].items()
                ):
                    bc[i].metric(k.replace('_',' ').title(), v)

            # Safety verdict
            st.subheader("Safety assessment")
            crit = nm_sum.get('by_level',{}).get('CRITICAL',0)
            high = nm_sum.get('by_level',{}).get('HIGH',0)
            min_ttc = nm_sum.get('min_ttc') or 999

            if crit > 50 or min_ttc < 0.5:
                st.error(
                    f"🔴 **HIGH RISK** — {crit} critical near-miss events "
                    f"detected. Minimum TTC: {min_ttc:.2f}s. "
                    f"This footage shows dangerous traffic conditions."
                )
            elif crit > 10 or high > 100:
                st.warning(
                    f"🟠 **ELEVATED RISK** — {crit} critical and "
                    f"{high} high-risk events detected. "
                    f"Traffic conditions require monitoring."
                )
            elif nm_sum.get('total_events',0) > 0:
                st.info(
                    f"🔵 **MODERATE** — Some near-miss events detected "
                    f"but no critical situations. "
                    f"Min TTC: {min_ttc:.2f}s."
                )
            else:
                st.success(
                    "🟢 **SAFE** — No significant incidents detected "
                    "in this footage."
                )

            # Recent incidents feed
            if all_nm_events or all_beh_events:
                st.subheader("Recent incidents (last 20)")
                recent = sorted(
                    all_nm_events[-10:] + all_beh_events[-10:],
                    key=lambda x: x.frame_idx, reverse=True
                )[:20]
                for ev in recent:
                    desc = getattr(ev, 'description', str(ev))
                    sev  = getattr(ev, 'severity', 2)
                    if sev >= 4 or 'NEAR MISS' in desc.upper():
                        st.markdown(
                            f'<div style="background:#FEE2E2;'
                            f'border-left:4px solid #E24B4A;'
                            f'padding:6px 12px;margin:3px 0;'
                            f'border-radius:4px;font-size:13px;'
                            f'color:#7B1111">'
                            f'Frame {ev.frame_idx}: {desc}</div>',
                            unsafe_allow_html=True
                        )
                    elif sev >= 3:
                        st.markdown(
                            f'<div style="background:#FEF3C7;'
                            f'border-left:4px solid #EF9F27;'
                            f'padding:6px 12px;margin:3px 0;'
                            f'border-radius:4px;font-size:13px;'
                            f'color:#633806">'
                            f'Frame {ev.frame_idx}: {desc}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div style="background:#F0FDF4;'
                            f'border-left:4px solid #639922;'
                            f'padding:6px 12px;margin:3px 0;'
                            f'border-radius:4px;font-size:13px;'
                            f'color:#27500A">'
                            f'Frame {ev.frame_idx}: {desc}</div>',
                            unsafe_allow_html=True
                        )

            # Download annotated video
            st.divider()
            st.subheader("Download")
            with open(out_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download annotated video",
                    data=f,
                    file_name=f"highway_safety_{uploaded.name}",
                    mime="video/mp4"
                )

        except Exception as e:
            status.error(f"Error during processing: {str(e)}")
            st.exception(e)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)