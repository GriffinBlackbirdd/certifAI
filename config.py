from pathlib import Path

# Base directory for course PDFs
PDF_DIR = Path("app/courses")

# Course configurations
COURSES = {
    "dp900": {
        "path": PDF_DIR / "test.pdf",
        "name": "Azure Data Fundamentals",
        "description": "Learn the fundamentals of database concepts in a cloud environment",
        "estimated_hours": 40
    },
    "ai900": {
        "path": PDF_DIR / "ai900.pdf",
        "name": "Azure AI Fundamentals",
        "description": "Master the basics of artificial intelligence in Azure",
        "estimated_hours": 35
    },
    # Add more courses as needed
}