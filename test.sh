# Run unit tests
pytest --cov-report term-missing --cov=apcmodels test_unit/

# Run acceptance tests
pytest test_acceptance/

# Run unit tests to generate coverage badge
coverage run -m pytest test_unit/
coverage-badge -o coverage.svg