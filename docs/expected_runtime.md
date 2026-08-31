# Expected Runtime

## Table Reproduction From Saved Artifacts

The default reproduction scripts read included aggregate CSVs, regenerate stable table CSVs, and verify exact equality. Expected runtime is seconds on a typical laptop.

## Fresh Model Inference

Fresh model inference is not part of the default public release workflow. Running the full protocol from raw prompts requires GPU hardware, local model access, and substantially longer runtime depending on model size. The release is designed so readers can verify reported tables without rerunning model inference.

## Bootstrap And Diagnostics

Saved bootstrap outputs are included. Recomputing bootstrap intervals from lower-level item artifacts may take minutes, depending on the number of bootstrap samples and local hardware.
