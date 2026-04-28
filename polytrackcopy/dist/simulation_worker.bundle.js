"use strict";

// Minimal offline stub for Kodub PolyTrack simulation worker (`Ik` matches main bundle).
//
// Main bundle enums (same order as shipped `9209-dist-main.bundle.js`):
// Init=0 CreateCar=1 StartCar=2 TestDeterminism=3 Update=4 DeterminismResult=5

const Ik = Object.freeze({
  Init: 0,
  CreateCar: 1,
  StartCar: 2,
  TestDeterminism: 3,
  Update: 4,
  DeterminismResult: 5,
});

self.onmessage = (evt) => {
  const msg = evt.data || {};
  const t = msg.messageType;

  if (t === Ik.TestDeterminism) {
    // Typos reproduced from main bundle: `isDeterminstic`.
    self.postMessage({
      messageType: Ik.DeterminismResult,
      isDeterminstic: true,
    });
    return;
  }

  // Do not ACK Init/CreateCar/StartCar/Update — those are only used for verifier
  // simulation (async `validate`) and are not required to finish the main-menu load.
};
