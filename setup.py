from setuptools import setup, find_packages

setup(
    name="s-space-navigation",
    version="0.4.0",
    description="Read, Navigate, and Control Transformer Internal Representations — with Thinking Mode Control",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.38",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
    },
)
