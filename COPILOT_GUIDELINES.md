#
# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.


## Error Handling and Nullability

- When writing a function or method that can fail or return no result, explicitly document this in the docstring and use Optional types in type hints (e.g., Optional[str]).
- If a function can return None (or a similar sentinel value), ensure all calling code checks for this case and handles it appropriately (e.g., with if result is None: ...).
- Prefer raising exceptions for truly exceptional or unrecoverable errors, and document which exceptions may be raised.
- When updating a function to allow None or to raise exceptions, always update all callers to handle the new behavior.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
   """
   Calculate the area of a circle given the radius.
    
   Parameters:
   radius (float): The radius of the circle.
    
   Returns:
   float: The area of the circle, calculated as π * radius^2.
   """
   import math
   return math.pi * radius ** 2
```
# Copilot Guidelines for RAG-Lab

These guidelines are for GitHub Copilot and any AI assistant working in this repository. Always follow these rules before making changes or generating code:

1. **Read Project Documentation**
   - Always check the following files for project context and requirements:
     - `README.md`
     - All files in the `docs/` directory (especially `timeline.md`, `implementation_plan.md`, etc.)
     - This `COPILOT_GUIDELINES.md` file

2. **Respect Project Plan and Timeline**
   - Ensure all code and suggestions align with the implementation plan and timeline described in the documentation.

3. **Metadata and Extraction**
   - Use the tools and approaches specified in the docs (e.g., `ffprobe`, `pymediainfo`, `pysubs2`, etc.) for metadata and data extraction.

4. **Consistency**
   - Follow naming conventions, data structures, and workflow as described in the documentation.

5. **Ask for Clarification**
   - If requirements are unclear or missing, prompt the user to clarify or update the documentation.

6. **Update Documentation**
   - When adding new features or making significant changes, update the relevant documentation files.

---

By following these guidelines, Copilot and other AI tools will help maintain project consistency and quality.
