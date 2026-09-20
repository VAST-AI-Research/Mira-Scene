# Example cases

This directory contains prepared Mira-Scene examples selected by
`test_data/filter.txt`.

```text
example_cases/
├── images/                 # Flat source-image directory used by the pipeline
│   └── <case>.png
└── cases/                  # Case root used by the Web UI and pipeline output
    └── <case>/
        ├── case.json
        ├── scene_graph.json
        └── input/
            ├── source.png
            ├── scene.png
            ├── preprocessing.json
            ├── mask_000.png ...
            ├── floor_mask.png
            └── scene_fg.png
```

`images/` preserves the original inputs. Each directory under `cases/`
contains the normalized image, prepared object and floor masks, foreground,
and scene graph needed to inspect the segmentation or continue later pipeline
stages. Generated depth, CCM, mesh, floor, scene, environment, log, review,
and manifest files are intentionally omitted.

See the inference guide for Web UI and pipeline commands:
[English](../infer_scripts/README.md#bundled-example-cases) |
[简体中文](../infer_scripts/README_zh-CN.md#仓库示例).
