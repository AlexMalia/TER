# Kaggle & Resume Guide

## Intégration Kaggle

### Prérequis

- CLI Kaggle installée et configurée (`~/.kaggle/kaggle.json`)
- Variable d'env `WANDB_API_KEY` ou clé dans `~/.netrc`
- Datasets créés sur Kaggle : `dino-code`, `dino-checkpoints`

### Setup initial (une seule fois)

```bash
cd DinoImpl
./kaggle/kaggle_manager.sh setup
```

Crée les datasets Kaggle et pousse le kernel.

### Cycle d'entraînement normal

```bash
# 1. Pousser le code et lancer l'entraînement
./kaggle/kaggle_manager.sh run

# 2. Vérifier le statut
./kaggle/kaggle_manager.sh status

# 3. Télécharger les outputs (checkpoints, logs, plots)
./kaggle/kaggle_manager.sh output

# 4. Pousser le checkpoint pour le prochain run
./kaggle/kaggle_manager.sh push-checkpoint
```

### Choisir le dataset

Dans `kaggle/kernel/train_kaggle.py`, ligne 22 :

```python
CONFIG_FILE = "kaggle-imagenette.yaml"   # ou "kaggle-imagenet100.yaml"
```

---

## Reprendre un entraînement

### Méthode 1 — Automatique via `checkpoint_latest.pth`

Mettre `resume_from: null` dans le YAML et lancer avec `--resume` :

```bash
python scripts/train.py --config configs/imagenette.yaml --resume
```

Détecte automatiquement le dernier checkpoint dans `checkpoint_dir`.

### Méthode 2 — Chemin explicite dans le YAML

```yaml
checkpoint_config:
  checkpoint_dir: ./outputs/checkpoints/...
  resume_from: ./outputs/checkpoints/.../checkpoint_latest.pth
```

### Méthode 3 — Argument CLI

```bash
python scripts/train.py --config configs/imagenette.yaml \
  --resume ./outputs/checkpoints/checkpoint_epoch_0050.pth
```

### Sur Kaggle (automatique)

Dans `kaggle/kernel/train_kaggle.py`, ligne 23 :

```python
RESUME_TRAINING = True   # False pour repartir de zéro
```

Si `True`, le script cherche `checkpoint_latest.pth` dans le dataset `dino-checkpoints`.

### Ce qui est restauré à la reprise

- Poids du student et du teacher
- État de l'optimizer
- Centre du loss DINO
- Epoch et itération courante
- Historique des métriques
- Run WandB (continuité de la courbe)

> **Note :** Le scheduler est recalculé par fast-forward (pas de `load_state_dict`) pour éviter les bugs de `SequentialLR`.

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `kaggle/kaggle_manager.sh` | CLI pour gérer Kaggle |
| `kaggle/kernel/train_kaggle.py` | Script kernel (lignes 21-23 : config/resume) |
| `scripts/train.py` | Script local (lignes 172-181 : logique resume) |
| `src/dino/utils/checkpoint.py` | `save_checkpoint()`, `load_checkpoint()` |
| `src/dino/training/trainer.py` | `resume_from_checkpoint()` (ligne 281) |
| `configs/kaggle-imagenette.yaml` | Config Kaggle ImageNette |
| `configs/kaggle-imagenet100.yaml` | Config Kaggle ImageNet100 |

## Vérification

- Après `run` : lancer `status` pour confirmer que le kernel démarre
- Sur WandB : vérifier que le run reprend avec le même ID (pas de nouveau run créé)
- Après `output` : vérifier que `./outputs/checkpoints/checkpoint_latest.pth` est présent localement
