import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const MAX_ACTIVE = 6;
const active = [];
const defaults = { azimuth: 0, elevation: 10, maxActive: MAX_ACTIVE };
const CAMERA_MOVE_KEYS = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight']);

export function configureViewer(options = {}) {
  defaults.azimuth = Number(options.azimuth ?? defaults.azimuth);
  defaults.elevation = Number(options.elevation ?? defaults.elevation);
  defaults.maxActive = Math.max(1, Number(options.max_active_3d ?? defaults.maxActive));
}

function disposeMaterial(material) {
  if (!material) return;
  for (const value of Object.values(material)) if (value?.isTexture) value.dispose();
  material.dispose?.();
}

function restoreIdleViewer(holder, field, message = '') {
  if (!holder?.isConnected) return;
  const toolbar = holder.querySelector('.viewer-controls');
  const button = document.createElement('button');
  button.className = 'load-3d';
  button.type = 'button';
  button.textContent = 'Load 3D';
  if (message) button.title = message;
  button.addEventListener('click', () => load3D(holder, field, button));
  const nodes = toolbar ? [toolbar, button] : [button];
  if (message) {
    const note = document.createElement('div');
    note.className = 'legend';
    note.textContent = message;
    nodes.push(note);
  }
  holder.replaceChildren(...nodes);
}

function dispose(state, message = '') {
  if (state.closed) return;
  state.closed = true;
  cancelAnimationFrame(state.frame);
  state.observer.disconnect();
  state.keyboard?.dispose();
  state.controls.dispose();
  state.root.traverse((item) => { item.geometry?.dispose(); if (Array.isArray(item.material)) item.material.forEach(disposeMaterial); else disposeMaterial(item.material); });
  state.environmentTexture?.dispose();
  state.renderer.dispose();
  state.renderer.forceContextLoss?.();
  const index = active.indexOf(state); if (index >= 0) active.splice(index, 1);
  if (message) restoreIdleViewer(state.holder, state.field, message);
}

export function disposeAllViewers() {
  for (const state of [...active]) dispose(state);
}

function disposeHolder(holder) {
  for (const state of [...active]) if (state.holder===holder) dispose(state);
}

function setEnvironmentRotation(holder, degrees) {
  const state=active.find(item=>item.holder===holder);
  if (!state) return;
  const radians=THREE.MathUtils.degToRad(Number(degrees)||0);
  if (state.scene.backgroundRotation) state.scene.backgroundRotation.y=radians;
  if (state.scene.environmentRotation) state.scene.environmentRotation.y=radians;
}

function frame(camera, controls, object, azimuth, elevation) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) throw new Error('3D asset has empty bounds');
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 1e-4);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * .5)) * 1.18;
  const az = THREE.MathUtils.degToRad(azimuth), el = THREE.MathUtils.degToRad(elevation);
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).add(new THREE.Vector3(distance*Math.cos(el)*Math.sin(az),distance*Math.sin(el),distance*Math.cos(el)*Math.cos(az)));
  camera.near=Math.max(radius/1000,1e-5); camera.far=Math.max(distance+radius*8,10); camera.updateProjectionMatrix(); camera.lookAt(sphere.center); controls.update();
  return radius;
}

function keyboardCameraControls(element) {
  const pressed=new Set();
  const keydown=(event)=>{if(!CAMERA_MOVE_KEYS.has(event.code))return;pressed.add(event.code);if(!event.code.startsWith('Shift'))event.preventDefault();};
  const keyup=(event)=>{if(!CAMERA_MOVE_KEYS.has(event.code))return;pressed.delete(event.code);if(!event.code.startsWith('Shift'))event.preventDefault();};
  const clear=()=>pressed.clear();
  const focus=()=>element.focus({preventScroll:true});
  element.tabIndex=0;
  element.setAttribute('aria-label','Interactive 3D scene. Use W and S to move forward and back, A and D left and right, Q and E down and up.');
  element.addEventListener('keydown',keydown);
  element.addEventListener('keyup',keyup);
  element.addEventListener('blur',clear);
  element.addEventListener('pointerdown',focus);
  return {pressed,dispose(){element.removeEventListener('keydown',keydown);element.removeEventListener('keyup',keyup);element.removeEventListener('blur',clear);element.removeEventListener('pointerdown',focus);pressed.clear();}};
}

function moveCamera(state, elapsed) {
  const keys=state.keyboard.pressed;
  if (!keys.size || elapsed<=0) return;
  const forward=new THREE.Vector3();
  state.camera.getWorldDirection(forward).normalize();
  const right=new THREE.Vector3().crossVectors(forward,state.camera.up).normalize();
  const up=new THREE.Vector3().copy(state.camera.up).normalize();
  const offset=new THREE.Vector3();
  if(keys.has('KeyW'))offset.add(forward);
  if(keys.has('KeyS'))offset.sub(forward);
  if(keys.has('KeyD'))offset.add(right);
  if(keys.has('KeyA'))offset.sub(right);
  if(keys.has('KeyE'))offset.add(up);
  if(keys.has('KeyQ'))offset.sub(up);
  if(!offset.lengthSq())return;
  const fast=keys.has('ShiftLeft')||keys.has('ShiftRight');
  offset.normalize().multiplyScalar(state.moveSpeed*elapsed*(fast?3:1));
  state.camera.position.add(offset);
  state.controls.target.add(offset);
}

export async function load3D(holder, field, button) {
  disposeHolder(holder);
  button.disabled=true; button.textContent='Loading…';
  const toolbar=holder.querySelector('.viewer-controls');
  let renderer=null, controls=null, root=null, environmentTexture=null;
  try {
    const width=Math.max(holder.clientWidth,240), height=Math.max(holder.clientHeight,220);
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:false}); renderer.setPixelRatio(Math.min(devicePixelRatio||1,2)); renderer.setSize(width,height,false); renderer.outputColorSpace=THREE.SRGBColorSpace;
    const scene=new THREE.Scene(); scene.background=new THREE.Color(0x070d17);
    const camera=new THREE.PerspectiveCamera(45,width/height,.01,1000);
    controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
    root=new THREE.Group(); scene.add(root);
    scene.add(new THREE.HemisphereLight(0xffffff,0x334466,2.2)); const light=new THREE.DirectionalLight(0xffffff,2.5); light.position.set(3,5,4); scene.add(light);
    if (field.kind === 'ply') {
      const geometry=await new PLYLoader().loadAsync(field.url); geometry.computeBoundingSphere();
      const material=new THREE.PointsMaterial({size:Number(field.point_size||.01),sizeAttenuation:true,vertexColors:geometry.hasAttribute('color'),color:geometry.hasAttribute('color')?0xffffff:(field.color||0xe95b5b)});
      root.add(new THREE.Points(geometry,material));
    } else {
      const gltf=await new GLTFLoader().loadAsync(field.url); root.add(gltf.scene);
    }
    if (field.environment_url) {
      try {
        environmentTexture=await new THREE.TextureLoader().loadAsync(field.environment_url);
        environmentTexture.mapping=THREE.EquirectangularReflectionMapping;
        environmentTexture.colorSpace=THREE.SRGBColorSpace;
        scene.background=environmentTexture;
        scene.environment=environmentTexture;
        const rotation=THREE.MathUtils.degToRad(Number(field.environment_rotation)||0);
        if (scene.backgroundRotation) scene.backgroundRotation.y=rotation;
        if (scene.environmentRotation) scene.environmentRotation.y=rotation;
      } catch (environmentError) {
        console.warn('Unable to load environment map',environmentError);
      }
    }
    if ((field.source_up||'y').toLowerCase()==='z') root.rotation.x=-Math.PI/2;
    holder.replaceChildren(renderer.domElement);
    if (toolbar) holder.append(toolbar);
    const radius=frame(camera,controls,root,Number(field.azimuth??defaults.azimuth),Number(field.elevation??defaults.elevation));
    const state={holder,field,renderer,scene,camera,controls,root,environmentTexture,observer:null,frame:0,closed:false,moveSpeed:Math.max(radius*1.2,1e-3),keyboard:keyboardCameraControls(renderer.domElement)};
    state.observer=new ResizeObserver(()=>{if(state.closed)return;const w=Math.max(holder.clientWidth,1),h=Math.max(holder.clientHeight,1);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}); state.observer.observe(holder);
    active.push(state); while(active.length>defaults.maxActive) dispose(active[0],`Viewer closed (maximum ${defaults.maxActive} active 3D views).`);
    renderer.domElement.focus({preventScroll:true});
    let previous=performance.now();
    const animate=(now)=>{if(state.closed||!holder.isConnected){dispose(state);return;}const elapsed=Math.min(Math.max((now-previous)/1000,0),.1);previous=now;moveCamera(state,elapsed);controls.update();renderer.render(scene,camera);state.frame=requestAnimationFrame(animate);}; state.frame=requestAnimationFrame(animate);
  } catch (error) {
    controls?.dispose();
    if (root) root.traverse((item)=>{item.geometry?.dispose();if(Array.isArray(item.material))item.material.forEach(disposeMaterial);else disposeMaterial(item.material);});
    environmentTexture?.dispose();
    renderer?.dispose(); renderer?.forceContextLoss?.();
    const note=document.createElement('div'); note.className='error'; note.textContent=`Unable to load 3D: ${error?.message||error}`;
    const retry=document.createElement('button'); retry.className='load-3d'; retry.type='button'; retry.textContent='Retry'; retry.addEventListener('click',()=>load3D(holder,field,retry)); holder.replaceChildren(note,retry);
    if (toolbar) holder.prepend(toolbar);
  }
}

function loadCroppedImage(media, field) {
  const image=new Image(); image.loading='lazy'; image.alt=field.label||field.kind||'visualization'; image.src=field.url;
  image.addEventListener('load',()=>{const sx=Math.floor(image.naturalWidth/2),sw=image.naturalWidth-sx;const canvas=document.createElement('canvas');canvas.width=sw;canvas.height=image.naturalHeight;canvas.setAttribute('aria-label',image.alt);canvas.getContext('2d').drawImage(image,sx,0,sw,image.naturalHeight,0,0,sw,image.naturalHeight);media.prepend(canvas);image.remove();});
  image.addEventListener('error',()=>{const note=document.createElement('div');note.className='error';note.textContent='Unable to load image';media.replaceChildren(note);}); media.append(image);
}

export function fieldCell(field) {
  const cell=document.createElement('div'); cell.className=`cell ${field.kind||''}`;
  const media=document.createElement('div'); media.className='cell-media'; cell.append(media);
  if (!field.url) { const empty=document.createElement('div'); empty.className='placeholder'; empty.textContent='not available'; media.append(empty); }
  else if (['glb','gltf','ply'].includes(field.kind)) { let current={...field};const toolbar=document.createElement('div');toolbar.className='viewer-controls';let picker=null;if(Array.isArray(field.options)&&field.options.length){picker=document.createElement('select');picker.className='scene-picker';picker.setAttribute('aria-label','3D scene variant');for(const option of field.options)picker.append(new Option(option.label,option.url));toolbar.append(picker);current.url=picker.value;}const environment=document.createElement('label');environment.className='environment-control';const environmentText=document.createElement('span');environmentText.textContent=field.environment_url?'Environment rotation':'No environment map';const rotation=document.createElement('input');rotation.type='range';rotation.min='-180';rotation.max='180';rotation.step='1';rotation.value=String(Number(field.environment_rotation)||0);rotation.disabled=!field.environment_url;rotation.setAttribute('aria-label','Environment map horizontal rotation in degrees');const rotationValue=document.createElement('output');rotationValue.textContent=`${rotation.value}°`;if(!field.environment_url)environment.classList.add('disabled');rotation.addEventListener('input',()=>{rotationValue.textContent=`${rotation.value}°`;current.environment_rotation=Number(rotation.value);setEnvironmentRotation(media,rotation.value);});environment.append(environmentText,rotation,rotationValue);toolbar.append(environment);const reset=()=>{disposeHolder(media);current={...field,url:picker?.value||field.url,environment_rotation:Number(rotation.value)};const next=document.createElement('button');next.className='load-3d';next.type='button';next.textContent='Load 3D';next.addEventListener('click',()=>load3D(media,current,next));media.replaceChildren(toolbar,next);};picker?.addEventListener('change',reset);media.append(toolbar);const button=document.createElement('button');button.className='load-3d';button.type='button';button.textContent='Load 3D';button.addEventListener('click',()=>load3D(media,current,button));media.append(button); }
  else if (field.crop==='right_half') loadCroppedImage(media,field);
  else { const image=document.createElement('img'); image.loading='lazy'; image.alt=field.label||field.kind||'visualization'; image.src=field.url; image.addEventListener('error',()=>{const note=document.createElement('div');note.className='error';note.textContent='Unable to load image';media.replaceChildren(note);}); media.append(image); }
  if (field.legend) { const legend=document.createElement('div');legend.className='legend';legend.textContent=field.legend;media.append(legend); }
  const caption=document.createElement('div');caption.className='cell-label';const label=document.createElement('span');label.textContent=field.label||field.kind||'Field';caption.append(label);if(field.note){const note=document.createElement('small');note.textContent=field.note;caption.append(note);}cell.append(caption); return cell;
}
