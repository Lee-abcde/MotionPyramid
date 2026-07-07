# MotionPyramid: Controllable Motion Synthesis via Stylized Phase Manifolds

**Jingyuan Li<sup>1</sup>, Peizhuo Li<sup>1</sup>, Andreas Aristidou<sup>2,3</sup>, Olga Sorkine-Hornung<sup>1</sup>**

<sup>1</sup>ETH Zurich, Switzerland  
<sup>2</sup>University of Cyprus, Cyprus  
<sup>3</sup>CYENS Centre of Excellence, Cyprus

**Computer Graphics Forum (Proceedings of SCA 2026)**

## Overview

This repository contains the implementation for **MotionPyramid**, a framework for controllable motion synthesis via stylized phase manifolds.

## Workflow

1. **Data processing in Unity**: import and preprocess motion data with the Unity tools described in [`Unity/README.md`](Unity/README.md).
2. **Python environment setup**: install the required Python dependencies and prepare the training environment following [`Python/README.md`](Python/README.md).
3. **VQ embedding training**: train the VQ phase embedding model with the Python module described in [`Python/README.md`](Python/README.md).
4. **Average-pose checking**: inspect the learned phase representation and average-pose behavior with the Unity tools described in [`Unity/README.md`](Unity/README.md).
5. **Text-to-phase and phase-to-motion training**: train the diffusion models for text-conditioned phase generation and phase-to-motion synthesis with the Python module described in [`Python/README.md`](Python/README.md).
6. **Inference**: run the trained Python models to generate motion results following [`Python/README.md`](Python/README.md).
7. **Visualization in Unity**: load and inspect the generated results with the Unity scenes and tools described in [`Unity/README.md`](Unity/README.md).

## Repository Structure

```text
MotionPyramid_private/
|-- Python/    # Python training and inference code
|-- Unity/     # Unity visualization and demo framework
`-- README.md  # Project overview
```

## Citation

If you use this code or project resources, please cite:

```bibtex
@article{li2026motionpyramid,
  title   = {MotionPyramid: Controllable Motion Synthesis via Stylized Phase Manifolds},
  author  = {Li, Jingyuan and Li, Peizhuo and Aristidou, Andreas and Sorkine-Hornung, Olga},
  journal = {Computer Graphics Forum},
  note    = {Proceedings of SCA 2026},
  year    = {2026}
}
```

## Acknowledgements

This work was supported in part by the European Research Council (ERC) under the European Union's Horizon 2020 Research and Innovation Programme (ERC Consolidator Grant, agreement No. 101003104, MYCLOTH) and by the EU Commission's Horizon Europe program (grant No. 101178362). Open access publishing was facilitated by ETH Zurich, as part of the Wiley-ETH Zurich agreement via the Consortium Of Swiss Academic Libraries.
