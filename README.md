# WVH Guide

A Python package for generating AI-assisted navigation instructions and visualizations inside the WVH building.

---

## 📦 Installation

Clone the repo and install the package locally:

```bash
git clone https://github.com/YOUR_USERNAME/wvh_guide_py.git
cd wvh_guide_py
pip install -e .
```

> Requires Python 3.9+, and an environment with `matplotlib`, `networkx`, and `google-generativeai`.

---

## 🚀 CLI Usage

After installation, use the built-in command-line tool:

```bash
wvh-guide --start f1_p01 --goal exit --api_key YOUR_API_KEY --model gemini-2.5-pro-exp-03-25
```

Options:

- `--start` and `--goal`: Node IDs or goal types (like `exit` or `elevator`)
- `--api_key`: Your Gemini API key (can also be set in `.env`)
- `--model`: Gemini model name (e.g. `gemini-2.5-pro-exp-03-25`)
- `--no-map`: Skip visualizing the map in a popup

---

## 🐍 Programmatic Use

You can also import and call the tool from Python:

```python
from wvh_guide import run

summary, path = run(
    api_key="YOUR_KEY",
    model="gemini-2.5-pro-exp-03-25",
    start="f1_p01",
    goal="exit",
    show_map=True
)
```

---

## 📁 Data

The package includes a preloaded map file (`WVH.json`) internally,
so users do not need to specify it explicitly.

---

## 🧪 Development

To run from source:

```bash
pip install -e .
```

To test:

```bash
python -m wvh_guide --start f1_p01 --goal exit --api_key YOUR_KEY --model gemini-2.5
```
