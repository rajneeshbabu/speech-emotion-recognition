from setuptools import setup, find_packages

setup(
    name="speech-emotion-recognition",
    version="1.0.0",
    author="Rajneesh",
    author_email="rajneeshb9458@gmail.com",
    description="Production-quality Speech Emotion Recognition with 6 ML models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rajneeshbabu/speech-emotion-recognition",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.30.0",
        "kaggle>=1.5.16",
    ],
    extras_require={
        "deep": ["tensorflow>=2.13.0"],
        "xgb": ["xgboost>=2.0.0"],
        "shap": ["shap>=0.42.0"],
        "all": [
            "tensorflow>=2.13.0",
            "xgboost>=2.0.0",
            "shap>=0.42.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ser-train=train:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
