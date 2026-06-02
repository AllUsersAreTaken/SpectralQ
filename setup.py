from setuptools import setup, find_packages

setup(
    name="spectralq",
    version="0.1.0",
    description="6-bit DCT quantization of LLM weights via a chunk-major CUDA kernel",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AllUsersAreTaken/SpectralQ",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "numpy>=1.24",
    ],
)
