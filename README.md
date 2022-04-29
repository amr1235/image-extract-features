# Feature extraction 
For given set of images (grayscale and color). we can extract the unique features in all images using Harris operator and λ-, generate feature descriptors using scale invariant features (SIFT), and matching the image set features using sum of squared differences (SSD) and normalized cross correlations.

---

# Setup
1. From the command line create a virtual environment and activate.
```sh
# Windows
> python -m venv .venv
> .venv\Scripts\activate

# Linux
> python3 -m venv .venv
> source .venv/bin/activate
```

2. Install the dependencies.
```sh
> pip install -r requirements.txt
```

3. Run the application.
```sh
> python APP.py
```