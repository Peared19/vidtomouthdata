import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export async function createRenderer(containerEl) {
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    45,
    containerEl.clientWidth / containerEl.clientHeight,
    0.1,
    100
  );
  camera.position.set(0, 0.1, 0.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  containerEl.appendChild(renderer.domElement);

  // Environment lighting for nicer default look.
  const environment = new RoomEnvironment();
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(environment).texture;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.1, 0);
  controls.update();

  function onResize() {
    camera.aspect = containerEl.clientWidth / containerEl.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
  }
  window.addEventListener('resize', onResize);

  const loader = new GLTFLoader();
  const ktx2Loader = new KTX2Loader()
    .setTranscoderPath('../node_modules/three/examples/jsm/libs/basis/')
    .detectSupport(renderer);
  loader.setKTX2Loader(ktx2Loader);
  loader.setMeshoptDecoder(MeshoptDecoder);

  let rootScene = null;
  let headMesh = null;

  async function loadModel(url) {
    const gltf = await loader.loadAsync(url);
    rootScene = gltf.scene;
    scene.add(rootScene);

    headMesh = null;
    rootScene.traverse((child) => {
      if (child.isMesh && child.morphTargetDictionary) {
        headMesh = child;
      }
    });

    if (!headMesh) {
      throw new Error('No mesh with morph targets found in GLB.');
    }

    return headMesh;
  }

  function startLoop(onTick) {
    renderer.setAnimationLoop((timeMs) => {
      controls.update();
      if (typeof onTick === 'function') onTick(timeMs);
      renderer.render(scene, camera);
    });
  }

  return {
    scene,
    camera,
    renderer,
    controls,
    loadModel,
    startLoop,
    getHeadMesh: () => headMesh
  };
}
