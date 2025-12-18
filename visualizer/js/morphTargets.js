// Builds a fast lookup table so per-frame updates are just numeric array writes.
// This is the biggest "feel" improvement (less jitter) because it avoids repeated string lookups.

export function buildMorphTargetIndex(headMesh, exampleFrame) {
  const dict = headMesh?.morphTargetDictionary;
  const totalTargets = dict ? Object.keys(dict).length : 0;

  if (!dict) {
    throw new Error('Head mesh has no morphTargetDictionary.');
  }

  const indicesByName = Object.create(null);
  const missingNames = [];

  const frameKeys = Object.keys(exampleFrame || {});

  for (const name of frameKeys) {
    let index = dict[name];
    if (index === undefined) {
      index = dict[`blendShape1.${name}`];
    }
    if (index === undefined) {
      index = dict[`BlendShape1.${name}`];
    }

    if (index === undefined) {
      missingNames.push(name);
      continue;
    }

    indicesByName[name] = index;
  }

  console.info(
    `[morph] Mapped ${Object.keys(indicesByName).length} / ${frameKeys.length} blendshape keys ` +
      `(mesh has ${totalTargets} morph targets)`
  );

  return { indicesByName, missingNames, totalTargets };
}

export function applyFrameFast(headMesh, morphIndex, frameData) {
  if (!headMesh || !morphIndex || !frameData) return;
  const influences = headMesh.morphTargetInfluences;
  if (!influences) return;

  // Only set keys that exist in morphIndex.
  for (const [name, value] of Object.entries(frameData)) {
    const idx = morphIndex[name];
    if (idx === undefined) continue;
    influences[idx] = value;
  }
}
