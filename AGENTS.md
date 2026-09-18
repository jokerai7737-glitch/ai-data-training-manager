# Vibe Coding Project Rules

## Intent

Build a real CLI AI-data pipeline while keeping the software understandable, testable, and maintainable.

## Rules

1. Keep CLI, business logic, and file I/O separated.
2. Prefer small functions with one responsibility.
3. Use meaningful names.
4. Avoid duplicated business knowledge.
5. Do not add unnecessary dependencies.
6. Do not place secrets or private data in the repository.
7. Review generated code before accepting it.
8. Run tests after meaningful changes.
9. Use descriptive Conventional Commit messages.
10. Do not change architecture without a clear reason.

## Acceptance Criteria

- CLI starts successfully.
- Dataset validation works.
- Preprocessing creates a clean dataset.
- Training creates a model artifact.
- Evaluation creates experiment results.
- Automated tests pass.
- The project can be reproduced from the README.
