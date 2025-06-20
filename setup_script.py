#!/usr/bin/env python3
"""
Setup script for Text-to-SQL Ollama Model
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-to-sql-ollama",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A complete pipeline for training and deploying text-to-SQL models with Ollama",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-to-sql-ollama",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
        ],
        "db": [
            "psycopg2-binary>=2.9.0",
            "mysql-connector-python>=8.0.0",
            "sqlparse>=0.4.4",
        ]
    },
    entry_points={
        "console_scripts": [
            "text-to-sql-train=text_to_sql_train:main",
            "text-to-sql-deploy=text_to_sql_deploy:deploy_main",
            "text-to-sql-test=text_to_sql_test:test_main",
        ],
    },
    include_package_data=True,
    package_data={
        "text_to_sql": [
            "configs/*.yaml",
            "templates/*.txt",
            "examples/*.json",
        ],
    },
)