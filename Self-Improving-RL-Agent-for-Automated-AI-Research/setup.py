from setuptools import setup, find_packages

setup(
    name="self-improving-rl-agent",
    version="0.1.0",
    description="A self-improving meta-reinforcement-learning agent for automated AI research.",
    author="Nidhal Zitouni",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
