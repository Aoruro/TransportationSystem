# Installation Guide

## Python Version

Use Python 3.8 to 3.12. Python 3.12 is recommended for this project.

On Windows, select Python 3.12 explicitly when multiple versions are installed:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -X utf8 quick_test.py
```

## Install Dependencies

Recommended:

```bash
python -m pip install -r requirements.txt
```

Manual installation:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib pytest
```

## Dependencies

| Library | Minimum Version | Purpose |
| --- | --- | --- |
| numpy | 1.21.0 | Numerical computing |
| pandas | 1.3.0 | Data handling |
| scipy | 1.7.0 | Statistical analysis |
| scikit-learn | 1.0.0 | Machine learning models |
| matplotlib | 3.4.0 | Visualization |
| pytest | 7.0.0 | Unit testing |

## Tkinter

Tkinter is included with standard Python installers. If `tkinter` is unavailable:

- Windows: reinstall Python using the official full installer.
- Linux: run `sudo apt-get install python3-tk`.
- macOS: use a Python distribution that includes Tk support.

## Verify Installation

```bash
python -X utf8 quick_test.py
python -X utf8 full_test.py
python -m unittest discover -s tests -v
```
