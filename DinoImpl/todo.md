## Completed ✅

- [x] Implementer dataset (CIFAR-100 et Imagenette)
- [x] backbone CNN (ResNet18)
- [x] Implémenter backbone + projection head
- [x] Instancier student et teacher et EMA
- [x] Implémenter augmentations et multi crop (2 global + 6 local)
- [x] Implémenter centering et sharpening dans la loss
- [x] Mettre en place la boucle d'entraînement
- [x] Checkpoints (save/load/resume)
- [x] Fix negative loss bug (was iterating over wrong variables)

## To Do 📝

- [ ] Metrics : Entropy, KNN accuracy, linear probing
- [ ] Learning rate scheduling (cosine with warmup)
- [ ] Multi-GPU / distributed training
- [ ] Mixed precision training
- [ ] TensorBoard logging
- [ ] Create standalone training scripts (.py files)
- [ ] Support for ViT backbone
- [ ] Pre-trained model weights

## Nice to Have 🎯

- [ ] Advanced augmentations (RandAugment, AutoAugment)
- [ ] Gradient accumulation for larger effective batch sizes
- [ ] Feature visualization (t-SNE, UMAP)
- [ ] Attention map visualization
- [ ] Benchmark on multiple datasets