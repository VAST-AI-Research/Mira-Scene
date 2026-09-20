# Automatic segmentation provenance

This package contains the Stage 0 automatic-segmentation orchestration adapted from
`scene-seg-and-layers` commit `944a1061f1505d901d97a2f6ef8e709d3b868962`.

The shared SAM3 model implementation and weights are not copied. They are loaded by
`segmentation.backends.sam3` and injected into this optional strategy. Keep this
snapshot internal until the upstream redistribution-license review is complete.
