import logging
import subprocess
from threading import Thread, Lock
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from urllib.parse import unquote
from pathlib import Path
import cv2
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from dataclasses import dataclass
import queue
from onvif import ONVIFCamera

app = FastAPI()

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StreamConfig:
    fps: int
    quality: int
    max_width: int

PROFILES = {
    "ultra_low": StreamConfig(fps=5, quality=40, max_width=480),
    "low": StreamConfig(fps=8, quality=50, max_width=640),
    "medium": StreamConfig(fps=10, quality=60, max_width=800),
    "high": StreamConfig(fps=15, quality=70, max_width=1280),
}

def auto_select_profile(num_cameras: int) -> StreamConfig:
    if num_cameras >= 80:
        return PROFILES["ultra_low"]
    elif num_cameras >= 40:
        return PROFILES["low"]
    elif num_cameras >= 20:
        return PROFILES["medium"]
    else:
        return PROFILES["high"]

# ============================================================================
# GLOBAL STATE
# ============================================================================

class CameraStream:
    def __init__(self, camera_id: str, rtsp_url: str, config: StreamConfig):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.config = config
        self.current_frame: Optional[bytes] = None
        self.frame_timestamp: float = 0
        self.frame_lock = Lock()
        self.active = True
        self.thread: Optional[Thread] = None
        self.frames_captured = 0
        self.errors = 0
        self.last_frame_time = 0
        
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.active = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info(f"Started {self.camera_id} @ {self.config.fps}FPS")
    
    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _capture_loop(self):
        """Optimized capture loop - FIXES LATENCY ISSUES"""
        cap = None
        reconnect_delay = 2
        frame_interval = 1.0 / self.config.fps
        consecutive_failures = 0
        max_failures = 5
        
        while self.active:
            try:
                if cap is None or not cap.isOpened():
                    logging.info(f"Connecting {self.camera_id}...")
                    
                    # CRITICAL: Use proper RTSP options for low latency
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    
                    # LATENCY FIX #1: Minimize buffer to get latest frame
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # LATENCY FIX #2: Set read timeout
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                    
                    # LATENCY FIX #3: Request specific FPS
                    cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                    
                    if not cap.isOpened():
                        logging.warning(f"{self.camera_id} failed, retry in {reconnect_delay}s")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 30)
                        continue
                    
                    logging.info(f"{self.camera_id} connected")
                    reconnect_delay = 2
                    consecutive_failures = 0
                
                frame_start = time.time()
                
                # LATENCY FIX #4: Grab multiple times to flush buffer
                # This ensures we're reading the LATEST frame, not old buffered frames
                for _ in range(2):
                    cap.grab()
                
                ret, frame = cap.retrieve()
                
                if not ret:
                    self.errors += 1
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        logging.error(f"{self.camera_id} too many failures, reconnecting")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                # Quick resize
                h, w = frame.shape[:2]
                if w > self.config.max_width:
                    scale = self.config.max_width / w
                    new_w = self.config.max_width
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Fast JPEG encode
                success, buffer = cv2.imencode(".jpg", frame, [
                    cv2.IMWRITE_JPEG_QUALITY, self.config.quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0  # Disable optimization for speed
                ])
                
                if success:
                    now = time.time()
                    with self.frame_lock:
                        self.current_frame = buffer.tobytes()
                        self.frame_timestamp = now
                        self.last_frame_time = now
                    self.frames_captured += 1
                
                # LATENCY FIX #5: Don't sleep if we're behind schedule
                elapsed = time.time() - frame_start
                sleep_time = frame_interval - elapsed
                
                # Only sleep if we're ahead of schedule
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.errors += 1
                logging.error(f"{self.camera_id} error: {e}")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1)
        
        if cap:
            cap.release()
        logging.info(f"{self.camera_id} stopped")
    
    def get_frame(self) -> Optional[bytes]:
        with self.frame_lock:
            # Check if frame is stale (older than 2 seconds)
            if self.current_frame and time.time() - self.frame_timestamp < 2.0:
                return self.current_frame
            return None
    
    def get_latency(self) -> float:
        """Get frame age in seconds"""
        with self.frame_lock:
            if self.frame_timestamp > 0:
                return time.time() - self.frame_timestamp
        return 0.0

streams: Dict[str, CameraStream] = {}
streams_lock = Lock()

RECORDING_DIR = Path("/app/recordings")
RECORDING_DIR.mkdir(parents=True, exist_ok=True)
active_recordings = {}

# ============================================================================
# LOGGING & MIDDLEWARE
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Low-Latency RTSP Streaming",
        "version": "2.1",
        "optimizations": [
            "Buffer flush mechanism",
            "Minimal buffering",
            "Fast frame grabbing",
            "Stale frame detection"
        ]
    }

@app.get("/start")
async def start_stream(
    rtspurl: str = Query(..., description="RTSP URL"),
    id: str = Query(..., description="Camera ID")
):
    rtsp_url = unquote(rtspurl)
    
    with streams_lock:
        if id in streams:
            logging.info(f"Camera {id} already streaming")
            return RedirectResponse(url=f"/view/{id}")
        
        config = auto_select_profile(len(streams) + 1)
        stream = CameraStream(id, rtsp_url, config)
        streams[id] = stream
        stream.start()
    
    # Give it a moment to connect
    time.sleep(0.5)
    
    logging.info(f"Started camera {id}")
    return RedirectResponse(url=f"/view/{id}")

@app.delete("/stop/{camera_id}")
async def stop_stream(camera_id: str):
    with streams_lock:
        stream = streams.pop(camera_id, None)
        if stream:
            stream.stop()
            return {"message": f"Stopped {camera_id}"}
    raise HTTPException(status_code=404, detail="Camera not found")

@app.get("/snapshot/{camera_id}")
async def get_snapshot(camera_id: str):
    with streams_lock:
        stream = streams.get(camera_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    frame = stream.get_frame()
    if not frame:
        raise HTTPException(status_code=503, detail="Frame not ready or stale")
    
    latency = stream.get_latency()
    
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Camera-ID": camera_id,
            "X-Frame-Latency-Ms": str(int(latency * 1000))
        }
    )

@app.get("/view/{camera_id}", response_class=HTMLResponse)
async def view_camera(camera_id: str):
    with streams_lock:
        stream = streams.get(camera_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    interval = int(1000 / stream.config.fps)
    
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{camera_id}</title>
    <style>
        body {{ margin: 0; background: #000; }}
        img {{ width: 100vw; height: 100vh; object-fit: contain; }}
        .info {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(0,0,0,0.9); color: #0f0;
            padding: 12px; border-radius: 5px;
            font-family: 'Courier New', monospace; font-size: 13px;
        }}
        .latency {{ color: #ff0; }}
        .latency.good {{ color: #0f0; }}
        .latency.bad {{ color: #f00; }}
    </style>
</head>
<body>
    <img id="stream">
    <div class="info" style="display:none;">
        <div>Camera: {camera_id}</div>
        <div>FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency" class="latency">0</span>ms</div>
    </div>
    <script>
        const img = document.getElementById('stream');
        const fpsEl = document.getElementById('fps');
        const latencyEl = document.getElementById('latency');
        let count = 0, lastTime = Date.now();
        let isLoading = false;
        
        function update() {{
            if (isLoading) return;
            
            isLoading = true;
            const startTime = performance.now();
            const newImg = new Image();
            
            newImg.onload = () => {{
                img.src = newImg.src;
                count++;
                
                const clientLatency = Math.round(performance.now() - startTime);
                const serverLatency = parseInt(newImg.getAttribute('data-latency') || '0');
                const totalLatency = clientLatency + serverLatency;
                
                latencyEl.textContent = totalLatency;
                latencyEl.className = 'latency ' + 
                    (totalLatency < 200 ? 'good' : totalLatency < 500 ? '' : 'bad');
                
                const now = Date.now();
                if (now - lastTime >= 1000) {{
                    fpsEl.textContent = count;
                    count = 0;
                    lastTime = now;
                }}
                
                isLoading = false;
            }};
            
            newImg.onerror = () => {{ 
                isLoading = false;
                console.error('Frame load failed');
            }};
            
            fetch('/snapshot/{camera_id}?t=' + Date.now())
                .then(response => {{
                    const latency = response.headers.get('X-Frame-Latency-Ms');
                    return response.blob().then(blob => {{
                        return {{ blob, latency }};
                    }});
                }})
                .then(data => {{
                    newImg.setAttribute('data-latency', data.latency || '0');
                    newImg.src = URL.createObjectURL(data.blob);
                }})
                .catch(err => {{
                    console.error(err);
                    isLoading = false;
                }});
        }}
        
        setInterval(update, {interval});
        update();
    </script>
</body>
</html>
    """)

@app.get("/view-grid", response_class=HTMLResponse)
async def view_grid(cameras: str = Query(..., description="Comma-separated IDs")):
    camera_ids = [c.strip() for c in cameras.split(",") if c.strip()]
    
    if not camera_ids:
        raise HTTPException(status_code=400, detail="No cameras")
    
    with streams_lock:
        valid = [cid for cid in camera_ids if cid in streams]
    
    if not valid:
        raise HTTPException(status_code=404, detail="No valid cameras")
    
    with streams_lock:
        fps = streams[valid[0]].config.fps
    interval = int(1000 / fps)
    
    grid = "".join([f'''
        <div class="cell">
            <div class="label">{cid}</div>
            <img data-id="{cid}">
            <div class="status" data-status="{cid}">●</div>
        </div>
    ''' for cid in valid])
    
    return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Grid - {len(valid)} Cameras</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #000; font-family: 'Courier New', monospace; }}
        .header {{
            background: #1a1a1a; color: #0f0; padding: 12px 20px;
            display: flex; justify-content: space-between;
            border-bottom: 2px solid #333;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 3px; padding: 3px;
            height: calc(100vh - 50px);
            overflow-y: auto;
        }}
        .cell {{
            position: relative; background: #111;
            aspect-ratio: 16/9; border: 1px solid #222;
        }}
        .cell img {{
            width: 100%; height: 100%; object-fit: contain;
        }}
        .label {{
            position: absolute; top: 5px; left: 5px;
            background: rgba(0,0,0,0.85); color: #0f0;
            padding: 4px 10px; font-size: 11px; 
            border-radius: 3px; z-index: 10;
        }}
        .status {{
            position: absolute; top: 5px; right: 5px;
            width: 10px; height: 10px; border-radius: 50%;
            background: #f00; z-index: 10;
            transition: background 0.3s;
        }}
        .status.active {{ 
            background: #0f0; 
            box-shadow: 0 0 10px #0f0;
        }}
        .stats {{ font-size: 13px; }}
        .stat-val {{ color: #0f0; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <div>{len(valid)} Cameras @ {fps} FPS</div>
        <div class="stats">
            FPS: <span class="stat-val" id="fps">0</span> | 
            Active: <span class="stat-val" id="active">0</span>/{len(valid)} |
            Avg Latency: <span class="stat-val" id="latency">0</span>ms
        </div>
    </div>
    <div class="grid">{grid}</div>
    <script>
        const cameras = {valid};
        const interval = {interval};
        const counters = {{}};
        const lastUpdate = {{}};
        const latencies = {{}};
        const loading = {{}};
        
        cameras.forEach(id => {{
            counters[id] = 0;
            lastUpdate[id] = Date.now();
            latencies[id] = [];
            loading[id] = false;
        }});
        
        function updateCam(id) {{
            if (loading[id]) return;
            
            loading[id] = true;
            const img = document.querySelector(`[data-id="${{id}}"]`);
            const status = document.querySelector(`[data-status="${{id}}"]`);
            if (!img) return;
            
            const startTime = performance.now();
            
            fetch('/snapshot/' + id + '?t=' + Date.now())
                .then(response => {{
                    const serverLatency = parseInt(response.headers.get('X-Frame-Latency-Ms') || '0');
                    return response.blob().then(blob => ({{ blob, serverLatency }}));
                }})
                .then(data => {{
                    const clientLatency = Math.round(performance.now() - startTime);
                    const totalLatency = clientLatency + data.serverLatency;
                    
                    latencies[id].push(totalLatency);
                    if (latencies[id].length > 10) latencies[id].shift();
                    
                    img.src = URL.createObjectURL(data.blob);
                    status.classList.add('active');
                    counters[id]++;
                    loading[id] = false;
                }})
                .catch(err => {{
                    status.classList.remove('active');
                    loading[id] = false;
                }});
        }}
        
        function updateStats() {{
            let totalFps = 0, active = 0, totalLatency = 0, latencyCount = 0;
            
            cameras.forEach(id => {{
                const now = Date.now();
                const elapsed = (now - lastUpdate[id]) / 1000;
                
                if (elapsed >= 1) {{
                    const fps = counters[id] / elapsed;
                    if (fps > 0) {{
                        totalFps += fps;
                        active++;
                    }}
                    counters[id] = 0;
                    lastUpdate[id] = now;
                }}
                
                if (latencies[id].length > 0) {{
                    const avgLat = latencies[id].reduce((a,b) => a+b) / latencies[id].length;
                    totalLatency += avgLat;
                    latencyCount++;
                }}
            }});
            
            document.getElementById('fps').textContent = 
                active > 0 ? Math.round(totalFps / active) : 0;
            document.getElementById('active').textContent = active;
            document.getElementById('latency').textContent = 
                latencyCount > 0 ? Math.round(totalLatency / latencyCount) : 0;
        }}
        
        // Stagger initial loads (50ms between cameras)
        cameras.forEach((id, i) => setTimeout(() => updateCam(id), i * 50));
        
        // Update all cameras
        setInterval(() => {{
            cameras.forEach(updateCam);
        }}, interval);
        
        // Update stats every second
        setInterval(updateStats, 1000);
    </script>
</body>
</html>
    """)

@app.get("/streams")
async def list_streams():
    with streams_lock:
        return {
            "total": len(streams),
            "cameras": [
                {
                    "id": cid,
                    "url": s.rtsp_url,
                    "fps": s.config.fps,
                    "quality": s.config.quality,
                    "frames": s.frames_captured,
                    "errors": s.errors,
                    "latency_ms": int(s.get_latency() * 1000)
                }
                for cid, s in streams.items()
            ]
        }

@app.delete("/stop-all")
async def stop_all():
    with streams_lock:
        for stream in list(streams.values()):
            stream.stop()
        streams.clear()
    return {"message": "All stopped"}

@app.get("/health")
async def health():
    with streams_lock:
        total = len(streams)
        frames = sum(s.frames_captured for s in streams.values())
        errors = sum(s.errors for s in streams.values())
        avg_latency = sum(s.get_latency() for s in streams.values()) / max(total, 1)
    
    return {
        "status": "ok",
        "cameras": total,
        "frames": frames,
        "errors": errors,
        "avg_latency_ms": int(avg_latency * 1000)
    }

# ============================================================================
# RECORDING
# ============================================================================

def record_loop(camera_id: str, rtsp_url: str, duration_min: int):
    while active_recordings.get(camera_id):
        ts = time.strftime("%Y%m%d_%H%M%S")
        date = time.strftime("%Y-%m-%d")
        outdir = RECORDING_DIR / camera_id / date
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{camera_id}_{ts}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-t", str(duration_min * 60),
            "-c:v", "copy",
            "-c:a", "aac",
            str(outfile)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            logging.error(f"Recording error {camera_id}: {e}")
            break
    
    active_recordings.pop(camera_id, None)

@app.post("/record/start")
async def start_recording(
    camera_id: str = Query(...),
    duration_min: int = Query(5, ge=1, le=60)
):
    with streams_lock:
        stream = streams.get(camera_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if camera_id in active_recordings:
        return {"message": "Already recording"}
    
    active_recordings[camera_id] = True
    Thread(
        target=record_loop,
        args=(camera_id, stream.rtsp_url, duration_min),
        daemon=True
    ).start()
    
    return {"message": f"Recording started: {camera_id}"}

@app.post("/record/stop")
async def stop_recording(camera_id: str):
    if camera_id not in active_recordings:
        raise HTTPException(status_code=404, detail="Not recording")
    
    active_recordings[camera_id] = False
    return {"message": f"Recording stopped: {camera_id}"}




@app.post("/discover_nvr_streams")
async def discover_nvr_streams(
    ip: str = Query(..., description="NVR IP address"),
    port: int = Query(80, description="NVR ONVIF port (default 80)"),
    username: str = Query(..., description="Username"),
    password: str = Query(..., description="Password")
):
    """
    Discover RTSP/ONVIF stream URLs from an NVR.
    Returns an array of RTSP URLs only.
    """
    try:
        # Connect to the NVR via ONVIF
        camera = ONVIFCamera(ip, port, username, password)
        media_service = camera.create_media_service()

        # Fetch all profiles
        profiles = media_service.GetProfiles()

        rtsp_urls = []
        for profile in profiles:
            try:
                stream_setup = {
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"}
                    },
                    "ProfileToken": profile.token
                }
                uri = media_service.GetStreamUri(stream_setup)

                # Build RTSP with credentials
                rtsp_url = uri.Uri.replace("rtsp://", f"rtsp://{username}:{password}@")
                rtsp_urls.append(rtsp_url)
            except Exception:
                continue

        return rtsp_urls

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONVIF discovery failed: {str(e)}")
    




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")







# <body>
#     <img id="stream">
#     <div class="info" style="display:none;">
#         <div>Camera: {camera_id}</div>
#         <div>FPS: <span id="fps">0</span></div>
#         <div>Latency: <span id="latency" class="latency">0</span>ms</div>
#     </div>