# AI Data & Training Manager

A command-line platform for managing datasets and running an AI training pipeline.

## Project Goal

The system demonstrates a structured AI data workflow using Python, Git/GitHub, testing, Clean Code, Linux/Bash automation, and Vibe Coding practices.

## Pipeline

Dataset ? Validation ? Cleaning ? Preprocessing ? Training ? Evaluation ? Model Artifact ? Results

## Main Commands

python -m ai_pipeline.cli.main --help

python -m ai_pipeline.cli.main validate data/sample/classification.csv

python -m ai_pipeline.cli.main pipeline run data/sample/classification.csv

## Architecture

- cli: command-line interface
- logic: business logic, models, validation, services
- io: CSV, JSON, and file operations
- tests: automated tests
- scripts: Linux/Bash automation

## Technologies

Python 3.12, argparse, CSV, JSON, pytest, Git, GitHub, Bash/Linux.

## Development Environment

The main development environment is Windows.
Kali Linux is used only for Linux-specific requirements such as Bash automation, permissions, processes, SSH, and scheduled tasks.
