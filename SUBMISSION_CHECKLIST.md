# Submission Checklist

## Before Running Experiments

- [ ] Add the verified upstream dataset URL, author, download date, and license
      to `DATASET.md`.
- [ ] Review the AI-assistance disclosure in `REFERENCES.md` against the course
      policy.
- [ ] Install dependencies with `py -3.12 -m pip install -r requirements.txt`.
- [ ] Run `py -3.12 scripts/build_small_dataset.py` if the source CSV changes.

## Code Verification

- [ ] Run `py -3.12 quick_test.py`.
- [ ] Run `py -3.12 full_test.py`.
- [ ] Run `py -3.12 -m unittest discover -s tests -v`.
- [ ] Launch `py -3.12 run_ui.py`.
- [ ] In the UI, test BFS, UCS, A*, and Learning A*.
- [ ] For Learning A*, click `Train ML Model` before starting the search.

## Report Artifacts

- [ ] Run `py -3.12 run_experiments.py`.
- [ ] Run `py -3.12 run_experiments.py --include-ml`.
- [ ] Select the clearest CSV metrics and PNG figures from `results/`.
- [ ] Explain that A* reduces expanded nodes substantially compared with BFS
      and UCS on the same 10-city instances.
- [ ] Report Learning A* results honestly, including prediction overhead when
      it does not outperform standard A*.
- [ ] Discuss factorial or exponential growth and why scale guards are needed.

## Required Submission Content

- [ ] Add project title, team names, and matriculation numbers to slides and
      report.
- [ ] Keep the presentation to 10 minutes plus 5 minutes for questions.
- [ ] Keep the report near 1500 words and include the word count.
- [ ] Include problem relevance, dataset, approach, results, interpretation,
      complexity, limitations, future work, conclusion, and references.
- [ ] Clearly present the UI, CSP, and ML work as above-and-beyond extensions.

## ZIP Cleanup

Do not include:

- `.git/`
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`
- Temporary experiment output that is not discussed in the report

Do include:

- Source code
- `README.md`, `INSTALLATION.md`, `DATASET.md`, and `REFERENCES.md`
- Both CSV datasets
- The specific figures or tables referenced by the report or presentation
