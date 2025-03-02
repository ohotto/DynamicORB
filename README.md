# DynamicORB

A simple experimental demo on using Grounding DINO + SAM to remove dynamic ORB feature points

## Submodules

Submodules need to be initialized and installed according to the source repository instructions

## Models

NEED to download models to path `models/`

```
models/
├── groundingdino_swint_ogc.pth
├── sam_vit_b_01ec64.pth
├── sam_vit_h_4b8939.pth
├── sam_vit_l_0b3195.pth
└── 
```

## Workfolder

The output files will be saved in `workfolder/{method}results/{INPUT_FILE_NAME}`

It is recommended to put the dataset files in `workfolder/` as well.
