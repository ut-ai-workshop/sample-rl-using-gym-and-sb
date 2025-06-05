# Sample RL using Gym and SB

## About

Sample Rainforcement learning using Gymnasium and StableBaseline

## Env

```bash
# install torch (gpu version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# install gym and sb
pip install gymnasium==1.0.0 stable-baselines3[extra]==2.6.0 sb3_contrib==2.6.0
```

## Usage

### Prepare settings

```bash
cp settings.sample.py settings.py
```

### Train

```bash
python train.py
```

## Plot training progress

```bash
python plot.py
```

## Test

```bash
python test.py
```
