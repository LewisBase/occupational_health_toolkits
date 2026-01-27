# 职业健康大数据智能决策系统 (OHTK) - AI Coding Agent Setup

## Project Overview
The Occupational Health Toolkit (OHTK) is a comprehensive framework for occupational health data analysis and intelligent decision-making. The system focuses on analyzing occupational diseases, particularly noise-induced hearing loss, and regional occupational disease outbreak trends.

## Key Components
- `staff_info`: Aggregates various information at the individual level
- `factory_info`: Consolidates factory environmental monitoring information at the factory level  
- `hazard_info`: Integrates detection information for various types of occupational hazards
- `detection_info`: Integrates direct results from various types of medical examinations
- `diagnose_info`: Performs further data processing and diagnosis based on examination information
- `model`: Contains various analytical models
- `utils`: Contains commonly used utilities

## Technical Stack
- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, pydantic, torch, matplotlib, seaborn, etc.
- Data modeling with Pydantic
- Logging with loguru
- Excel processing capabilities

## Code Style & Conventions
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Leverage Pydantic for data validation and serialization
- Use logging instead of print statements
- Write unit tests for all major functionality
- Include docstrings for classes and functions

## Important Files
- `setup.py`: Package setup and dependencies
- `requirements.txt`: Project dependencies
- `ohtk/__init__.py`: Main package initialization
- `README.md`: Project documentation

## Testing
- Tests are located in the `tests/` directory
- Use pytest for running tests
- New features should include corresponding test cases

## Development Guidelines
- Maintain backward compatibility when possible
- Follow the existing architectural patterns
- Document new constants in the appropriate constants modules
- Validate data inputs using Pydantic models
- Handle errors gracefully with appropriate logging

## Security Considerations
- Sanitize all external data inputs
- Validate file types and contents when processing Excel files
- Follow secure coding practices when handling user data
- Protect sensitive health information appropriately

## Common Tasks
- Add new hazard types by extending the BaseHazard class
- Create new detection result types by extending base detection classes
- Add new diagnostic capabilities in the diagnose_info module
- Extend staff information by modifying the StaffInfo class
- Build new analytical models in the model module