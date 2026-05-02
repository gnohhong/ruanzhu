# Project instructions

This project is a near-infrared spectroscopy tea withering moisture detection system.

## Project structure

- `nir/prediction`: Python algorithm, training, prediction, and model files.
- `nir/prediction/models`: saved model files, including ANN, Elman, TNet, PLSR, and LSSVR.
- `backend`: backend API service. It will call the Python prediction code later.
- `frontend`: React + TypeScript frontend interface.

## Frontend goal

Build a modern dashboard for the tea withering moisture detection system.

The frontend should include:

- Near-infrared spectrum data upload area
- Spectrum curve visualization
- Moisture prediction result card
- Withering status or stage display
- Model selection: ANN, Elman, TNet, PLSR, LSSVR
- Prediction history table
- Experiment parameter area
- Clean agricultural research system style

## Rules

- Work inside `frontend` for frontend tasks.
- Use React + TypeScript.
- Do not put `.pth` or `.pkl` model files into the frontend.
- The frontend should call backend APIs later, not load model files directly.
- Do not modify files in `nir/prediction` unless explicitly requested.
- When building or redesigning UI, use the `frontend-design` skill.

## Commands

Frontend:

```bash
cd frontend
npm install
npm run dev
npm run build