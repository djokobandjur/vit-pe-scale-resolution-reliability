# LAYER_STATUS_POLICY_v1.0

## Primary layer
**L4 remains the primary monitoring layer.**

The completed controlled experiments provide a functional justification: among the 12 prespecified secondary layers, L4 is the **only** layer at which the prespecified `higher attention entropy = more OOD` orientation is non-inverted (AUROC >= 0.5) for **both directions**, all 24 PE×seed cells, and both ViT-B and ViT-S. This uniqueness is a post-analysis characterization of the prespecified secondary profile; it does not replace the already locked L4 primary endpoint.

## L2 status
All 12 layers were part of the prespecified secondary layer profile. The specific L2 sign-inversion pattern and its interpretation were identified after observing that profile and are therefore reported as secondary/post-hoc characterization, not as a replacement primary endpoint.

## RoPE at L2
RoPE is the only PE family that preserves the expected F→C score orientation at L2 in both architectures (0/6 inverted seeds in each). Learned, Sinusoidal, and ALiBi each show 6/6 F→C inversions at L2 in both architectures.
