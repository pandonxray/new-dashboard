# New Dashboard

`New Dashboard` is a standalone trading dashboard project extracted from the original `Excel-` workflow and tuned for daily research use.

## What it includes

- Preset combinations kept as high-frequency entry points
- Custom combination analysis, decomposition, risk, and seasonality views
- Basis and downstream profit monitoring
- Softer workspace-style UI with clearer chart annotations
- Lunar-calendar seasonality view for China-specific seasonal reading

## Project structure

- `src/`: dashboard app, launcher, and analysis engines
- `config/`: app and metric configuration
- `assets/`: icon assets
- `scripts/`: build and desktop shortcut scripts
- `tests/`: test files

## Run locally

Use the batch launcher:

```bat
start_dashboard.bat
```

The packaged desktop app prefers port `8511` and will automatically move to the next available local port if it is already in use.

## Build the Windows executable

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```
