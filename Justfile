# Remove artifacts and caches
clean:
  find . -name '__pycache__' -type d -exec rm -rf {} +
  rm -rf .ruff_cache

# Run linter and type checker
check TARGET=".":
  uv run ruff check {{TARGET}}
  uv run ty check {{TARGET}}

# Fix lint errors and format code
format TARGET=".":
  uv run ruff check --fix {{TARGET}}
  uv run ruff format {{TARGET}}
