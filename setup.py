from setuptools import setup, find_packages

setup(
    name="ai-cache",
    version="0.1.0",
    description="Zero-overhead AI response caching with payload-aware keying",
    author="my-ai-stack",
    url="https://github.com/my-ai-stack/ai-cache",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "redis",  # optional, for Redis backend
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
