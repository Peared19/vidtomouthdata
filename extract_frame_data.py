"""
Frame adat kinyerés és HTML export
Egy konkrét frame-et kiinyerünk a videóból, exportáljuk a blend shape értékeket,
és létrehozunk egy HTML oldalt amely a 3D modellen megmutatja az arcállást
"""

import os
import cv2
import json
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
from frame_processor import process_frame_full_mouth

# -------------------- Beállítások --------------------
VIDEO_BASE = "D:/MestInt/datasets/gridcorpus/video"
ALIGN_BASE = "D:/MestInt/datasets/gridcorpus/align"
MODEL_PATH = "face_landmarker.task"

# -------------------- FaceLandmarker inicializálása --------------------
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True
)
landmarker = vision.FaceLandmarker.create_from_options(options)

def parse_align_file(align_path, sample_rate=25000):
    """Betölti az align fájlt"""
    word_list = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_sample = float(parts[0])
                end_sample = float(parts[1])
                word = parts[2]
                start_time_s = start_sample / sample_rate
                end_time_s = end_sample / sample_rate
                word_list.append((word, start_time_s, end_time_s))
    return word_list

def extract_first_non_sil_frame():
    """Lekéri az első nem-sil frame adatait"""
    
    speakers = sorted([s for s in os.listdir(VIDEO_BASE) 
                      if os.path.isdir(os.path.join(VIDEO_BASE, s))])
    
    speaker = speakers[0]
    speaker_video_path = os.path.join(VIDEO_BASE, speaker, speaker)
    speaker_align_path = os.path.join(ALIGN_BASE, speaker, "align")
    
    videos = sorted([v for v in os.listdir(speaker_video_path) 
                    if v.lower().endswith((".mpg", ".mp4"))])
    
    video_file = videos[0]
    video_path = os.path.join(speaker_video_path, video_file)
    align_file_name = os.path.splitext(video_file)[0] + ".align"
    align_path = os.path.join(speaker_align_path, align_file_name)
    
    word_list = parse_align_file(align_path)
    
    # Video betöltése
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    found_frame = False
    frame_data = None
    
    print(f"\n{'='*80}")
    print(f"🔍 Első nem-sil frame keresése...")
    print(f"{'='*80}\n")
    
    while not found_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        word_for_frame = None
        
        for word, start_time, end_time in word_list:
            if start_time <= current_time <= end_time:
                word_for_frame = word
                break
        
        if word_for_frame is not None and word_for_frame != "sil":
            mouth_data = process_frame_full_mouth(frame, landmarker)
            
            if mouth_data is not None:
                frame_data = {
                    "speaker": speaker,
                    "video": video_file,
                    "frame_idx": frame_idx,
                    "word": word_for_frame,
                    "fps": fps,
                    "timestamp": current_time,
                    "mouth_data": mouth_data
                }
                found_frame = True
                
                print(f"✅ Megtalálva!")
                print(f"   Frame: #{frame_idx}")
                print(f"   Szó: '{word_for_frame}'")
                print(f"   Idő: {current_time:.2f}s")
                print(f"   FPS: {fps}")
                print(f"   Speaker: {speaker}")
                print(f"   Video: {video_file}\n")
        
        frame_idx += 1
    
    cap.release()
    return frame_data

def create_html_viewer(frame_data, output_file="viewer.html"):
    """Létrehozza az interaktív HTML viewert"""
    
    blend_shapes = frame_data["mouth_data"]["blend_shapes"]
    
    # Blend shapes JSON stringként
    blend_shapes_json = json.dumps(blend_shapes)
    
    # Csak az aktív blend shapes (>0.01)
    active_shapes = {k: v for k, v in blend_shapes.items() if v > 0.01}
    active_shapes_json = json.dumps(active_shapes)
    
    frame_info = {
        "speaker": frame_data["speaker"],
        "video": frame_data["video"],
        "frame_idx": frame_data["frame_idx"],
        "word": frame_data["word"],
        "timestamp": frame_data["timestamp"]
    }
    frame_info_json = json.dumps(frame_info)
    
    html_content = f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Arc Animáció - Frame #{frame_data['frame_idx']} ({frame_data['word']})</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }}
        
        #canvas {{
            flex: 1;
            display: block;
        }}
        
        #sidebar {{
            width: 350px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            overflow-y: auto;
            padding: 20px;
            border-left: 2px solid #667eea;
        }}
        
        #sidebar h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        #sidebar h3 {{
            color: #aaa;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }}
        
        .frame-info {{
            background: rgba(102, 126, 234, 0.2);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 13px;
            line-height: 1.6;
        }}
        
        .frame-info label {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .frame-info div {{
            margin: 5px 0;
        }}
        
        .blend-shape-item {{
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }}
        
        .blend-shape-name {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #fff;
            font-size: 12px;
        }}
        
        .blend-shape-bar {{
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .blend-shape-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 3px;
            transition: width 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 5px;
            color: white;
            font-size: 10px;
            font-weight: bold;
        }}
        
        .controls {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .control-item {{
            margin: 12px 0;
        }}
        
        .control-item label {{
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
        }}
        
        .control-item input[type="range"] {{
            width: 100%;
            cursor: pointer;
        }}
        
        button {{
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            transition: background 0.3s;
        }}
        
        button:hover {{
            background: #764ba2;
        }}
        
        .warning {{
            color: #ffaa33;
            font-size: 12px;
            margin-top: 10px;
            padding: 10px;
            background: rgba(255, 170, 51, 0.1);
            border-left: 2px solid #ffaa33;
            border-radius: 3px;
        }}
        
        #scrollArea {{
            flex: 1;
        }}
    </style>
</head>
<body>
    <div id="canvas"></div>
    <div id="sidebar">
        <h2>🎭 Arcállásfelvétel</h2>
        
        <div class="frame-info">
            <div><label>Frame:</label> #{frame_data['frame_idx']}</div>
            <div><label>Szó:</label> <span style="color: #667eea; font-size: 14px; font-weight: bold;">{frame_data['word']}</span></div>
            <div><label>Idő:</label> {frame_data['timestamp']:.3f}s</div>
            <div><label>Speaker:</label> {frame_data['speaker']}</div>
            <div><label>Videó:</label> {frame_data['video']}</div>
        </div>
        
        <h3>📊 Aktív Arcállások</h3>
        <div id="blendShapesList"></div>
        
        <h3>🎮 Kontrollok</h3>
        <div class="controls">
            <div class="control-item">
                <label>Fejforgás X (Pitch)</label>
                <input type="range" id="rotationX" min="-90" max="90" value="0" step="1">
            </div>
            <div class="control-item">
                <label>Fejforgás Y (Yaw)</label>
                <input type="range" id="rotationY" min="-90" max="90" value="0" step="1">
            </div>
            <div class="control-item">
                <label>Fejforgás Z (Roll)</label>
                <input type="range" id="rotationZ" min="-90" max="90" value="0" step="1">
            </div>
            <button onclick="resetRotation()">Visszaállítás</button>
            <button onclick="resetAllInfluences()">Összes Arcállás Törlése</button>
        </div>
        
        <div class="warning">
            💡 Csúsztasd a szliderek gombokat az arcállások módosításához!
        </div>
    </div>

    <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@r128/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@r128/examples/jsm/"
            }}
        }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
        import {{ RoomEnvironment }} from 'three/addons/environments/RoomEnvironment.js';

        let scene, camera, renderer, mesh, controls;
        
        const BLEND_SHAPES = {blend_shapes_json};
        const ACTIVE_SHAPES = {active_shapes_json};
        
        console.log("🎯 Blend Shapes betöltve:", BLEND_SHAPES);
        console.log("✅ Aktív Arcállások:", ACTIVE_SHAPES);

        function init() {{
            // Scene
            const container = document.getElementById('canvas');
            scene = new THREE.Scene();
            
            // Camera
            camera = new THREE.PerspectiveCamera(
                45,
                container.clientWidth / window.innerHeight,
                1,
                20
            );
            camera.position.set(-1.8, 0.8, 3);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(container.clientWidth, window.innerHeight);
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1;
            renderer.setClearColor(0x666666, 1);
            container.appendChild(renderer.domElement);

            // Háttér
            const environment = new RoomEnvironment();
            const pmremGenerator = new THREE.PMREMGenerator(renderer);
            scene.environment = pmremGenerator.fromScene(environment).texture;

            // Modell betöltése
            const loader = new GLTFLoader();
            loader.load('models/facecap.glb', (gltf) => {{
                mesh = gltf.scene.children[0];
                scene.add(mesh);

                // Blend shapes alkalmazása
                const head = mesh.getObjectByName('mesh_2');
                if (head && head.morphTargetInfluences) {{
                    applyBlendShapes(head, BLEND_SHAPES);
                    createBlendShapeControls(head);
                }}
            }});

            // OrbitControls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.minDistance = 2.5;
            controls.maxDistance = 5;
            controls.minAzimuthAngle = -Math.PI / 2;
            controls.maxAzimuthAngle = Math.PI / 2;
            controls.maxPolarAngle = Math.PI / 1.8;
            controls.target.set(0, 0.15, -0.2);

            // Animation loop
            renderer.setAnimationLoop(animate);
            window.addEventListener('resize', onWindowResize);
        }}

        function applyBlendShapes(head, blendShapes) {{
            if (!head.morphTargetDictionary) {{
                console.error("❌ Morph target dictionary nem elérhető!");
                return;
            }}

            let appliedCount = 0;
            for (const [key, value] of Object.entries(head.morphTargetDictionary)) {{
                const cleanKey = key.replace('blendShape1.', '');
                const blendValue = blendShapes[cleanKey];
                
                if (blendValue !== undefined) {{
                    head.morphTargetInfluences[value] = blendValue;
                    appliedCount++;
                }}
            }}
            
            console.log(`✅ {{appliedCount}} arcállás alkalmazva!`);
        }}

        function createBlendShapeControls(head) {{
            const listDiv = document.getElementById('blendShapesList');
            listDiv.innerHTML = '';

            const shapeEntries = Object.entries(ACTIVE_SHAPES)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15);  // Top 15

            shapeEntries.forEach(([key, value]) => {{
                const item = document.createElement('div');
                item.className = 'blend-shape-item';
                
                const name = document.createElement('div');
                name.className = 'blend-shape-name';
                name.textContent = key;
                
                const barContainer = document.createElement('div');
                barContainer.className = 'blend-shape-bar';
                
                const bar = document.createElement('div');
                bar.className = 'blend-shape-fill';
                bar.style.width = (value * 100) + '%';
                bar.textContent = (value * 100).toFixed(1) + '%';
                
                barContainer.appendChild(bar);
                item.appendChild(name);
                item.appendChild(barContainer);
                
                // Slider
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '0';
                slider.max = '1';
                slider.step = '0.01';
                slider.value = value;
                slider.style.marginTop = '5px';
                slider.style.width = '100%';
                slider.style.cursor = 'pointer';
                
                slider.addEventListener('input', (e) => {{
                    const cleanKey = key.replace('blendShape1.', '');
                    const dictIndex = head.morphTargetDictionary[key];
                    head.morphTargetInfluences[dictIndex] = parseFloat(e.target.value);
                    bar.style.width = (parseFloat(e.target.value) * 100) + '%';
                    bar.textContent = (parseFloat(e.target.value) * 100).toFixed(1) + '%';
                }});
                
                item.appendChild(slider);
                listDiv.appendChild(item);
            }});
        }}

        function animate() {{
            controls.update();
            
            // Fejforgás
            const rotX = parseFloat(document.getElementById('rotationX').value) * Math.PI / 180;
            const rotY = parseFloat(document.getElementById('rotationY').value) * Math.PI / 180;
            const rotZ = parseFloat(document.getElementById('rotationZ').value) * Math.PI / 180;
            
            if (mesh) {{
                mesh.rotation.order = 'YXZ';
                mesh.rotation.x = rotX;
                mesh.rotation.y = rotY;
                mesh.rotation.z = rotZ;
            }}
            
            renderer.render(scene, camera);
        }}

        function onWindowResize() {{
            const container = document.getElementById('canvas');
            camera.aspect = container.clientWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, window.innerHeight);
        }}

        function resetRotation() {{
            document.getElementById('rotationX').value = 0;
            document.getElementById('rotationY').value = 0;
            document.getElementById('rotationZ').value = 0;
        }}

        window.resetAllInfluences = function() {{
            // Összes morph target nullázása
            console.log("🔄 Arcállások visszaállítása...");
            location.reload();
        }};

        init();
    </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_file

# ========== MAIN ==========
if __name__ == "__main__":
    print("\n" + "="*80)
    print("📹 FRAME ADAT KINYERÉS ÉS 3D EXPORT")
    print("="*80)
    
    # Frame adat kinyerése
    frame_data = extract_first_non_sil_frame()
    
    if frame_data:
        print(f"\n✅ Frame data kiinyerve!")
        print(f"\nBlend shapes:")
        blend_shapes = frame_data["mouth_data"]["blend_shapes"]
        active = {k: v for k, v in blend_shapes.items() if v > 0.01}
        for k, v in sorted(active.items(), key=lambda x: x[1], reverse=True):
            print(f"   {k}: {v:.4f}")
        
        # HTML exportálás
        html_file = create_html_viewer(frame_data)
        print(f"\n✅ HTML viewer létrehozva: {html_file}")
        print(f"   Nyisd meg a böngészőben: file:///{os.path.abspath(html_file)}")
        
        print("\n" + "="*80)
        print("🎯 Következő lépések:")
        print("="*80)
        print("1. Nyisd meg az HTML fájlt a böngészőben")
        print("2. Csúsztasd meg a szlidereket az arcállások módosításához")
        print("3. A fejforgáshoz használd a jobb panel kontrolljait")
        print("="*80 + "\n")
    else:
        print("❌ Nem sikerült frame adatot kinyerni!")
    
    landmarker.close()
