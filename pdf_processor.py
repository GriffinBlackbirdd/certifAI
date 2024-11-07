# pdf_processor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import asyncio
import logging
from typing import Dict, List, AsyncGenerator
import json
from datetime import datetime
import os
from pathlib import Path

def setup_logger(course_code: str) -> logging.Logger:
    """Set up logging configuration for the course processor"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(f"course_processor_{course_code}")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers = []

    # Create a unique log file for this processing run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{course_code}_{timestamp}.log"

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class CourseProcessor:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        self.logger = None
        self.processing_tasks = {}  # Store background processing tasks

        # Create debug directory for output inspection
        self.debug_dir = Path("debug_output")
        self.debug_dir.mkdir(exist_ok=True)

    def save_debug_output(self, course_code: str, stage: str, content: str):
        """Save debug output to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{course_code}_{stage}_{timestamp}.txt"
        filepath = self.debug_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Stage: {stage}\n")
            f.write("="*50 + "\n")
            f.write(content)

        return filepath

    async def process_pdf(self, pdf_path: str, style_prompt: str) -> AsyncGenerator[Dict, None]:
        """Process PDF with quick first module delivery"""
        course_code = Path(pdf_path).stem
        self.logger = setup_logger(course_code)

        try:
            self.logger.info(f"Starting to process PDF for course: {course_code}")
            self.save_debug_output(course_code, "style_prompt", style_prompt)

            # Load PDF
            self.logger.info("Loading PDF document")
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Split into introduction/first module and rest
            all_content = [page.page_content for page in pages]
            first_module = all_content[:2]  # First few pages for quick processing
            remaining_content = all_content[2:]

            # Save initial content for debugging
            self.save_debug_output(course_code, "first_module_raw", "\n".join(first_module))

            # Process first module quickly
            self.logger.info("Processing first module")
            first_module_processed = await self.process_quick_module(
                first_module,
                style_prompt
            )

            # Save processed first module
            self.save_debug_output(course_code, "first_module_processed", first_module_processed)

            # Generate first module components
            self.logger.info("Generating first module components")
            try:
                summary = await self._generate_summary(first_module_processed)
                key_points = await self._generate_key_points(first_module_processed)
                quiz = await self._generate_quiz(first_module_processed)
            except Exception as e:
                self.logger.error(f"Error generating module components: {str(e)}")
                summary = "Summary will be available soon."
                key_points = ["Key points being processed..."]
                quiz = self._generate_fallback_quiz()

            # Create and return first module
            first_module_data = {
                "status": "partial",
                "content": first_module_processed,
                "modules": [{
                    "number": 1,
                    "title": "Introduction to the Course",
                    "content": first_module_processed,
                    "summary": summary,
                    "key_points": key_points,
                    "quiz": quiz
                }],
                "message": "First module ready, processing remaining content..."
            }

            self.logger.info("First module processing completed")
            yield first_module_data

            # Start background processing for remaining content
            self.logger.info("Starting background processing")
            self.processing_tasks[course_code] = asyncio.create_task(
                self.process_remaining_content(remaining_content, style_prompt, course_code)
            )

            yield {
                "status": "background_started",
                "message": "Remaining modules are being processed in background"
            }

        except Exception as e:
            self.logger.error(f"Error in PDF processing: {str(e)}")
            yield {"status": "error", "error": str(e)}

    async def process_quick_module(self, content: list, style_prompt: str) -> str:
        """Process first module with minimal transformations"""
        try:
            quick_prompt = f"""
            Transform this introductory content while maintaining technical accuracy.
            Style: {style_prompt}

            Guidelines:
            1. Keep technical terms intact
            2. Maintain educational value
            3. Add engaging elements following the style
            4. Focus on clarity and understanding

            Content: {' '.join(content)}
            """
            response = await self.llm.ainvoke(quick_prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error in quick processing: {str(e)}")
            return ' '.join(content)  # Return original content if processing fails

    async def process_remaining_content(self, content: list, style_prompt: str, course_code: str):
        """Process remaining content in background"""
        try:
            self.logger.info(f"Starting background processing for {course_code}")
            processed_modules = []
            chunks = self.text_splitter.split_text(' '.join(content))

            for i, chunk in enumerate(chunks):
                self.logger.info(f"Processing chunk {i+1} of {len(chunks)}")
                try:
                    # Process chunk
                    response = await self.llm.ainvoke(self.create_prompt(chunk, style_prompt))
                    self.save_debug_output(course_code, f"module_{i+2}_content", response)

                    # Create module
                    module = {
                        "number": i + 2,  # Start from module 2
                        "title": await self._generate_module_title(response),
                        "content": response,
                        "summary": await self._generate_summary(response),
                        "key_points": await self._generate_key_points(response),
                        "quiz": await self._generate_quiz(response)
                    }

                    self.save_debug_output(
                        course_code,
                        f"module_{i+2}_complete",
                        json.dumps(module, indent=2)
                    )

                    processed_modules.append(module)

                    # Update database with new module
                    await self.update_course_content(course_code, module)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in background processing: {str(e)}")

    async def get_processing_status(self, course_code: str) -> dict:
        """Get the current processing status"""
        if course_code in self.processing_tasks:
            task = self.processing_tasks[course_code]
            if task.done():
                return {"status": "completed"}
            return {"status": "processing"}
        return {"status": "not_found"}

    async def _generate_module_title(self, content: str) -> str:
        """Generate a title for the module"""
        try:
            prompt = "Create a concise title (max 8 words) for this content:\n{content}\nTitle:"
            response = await self.llm.ainvoke(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            return "Module Title"

    async def _generate_summary(self, content: str) -> str:
        """Generate a summary of the module content"""
        try:
            prompt = f"Create a concise summary (2-3 sentences) of:\n{content}\nSummary:"
            response = await self.llm.ainvoke(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "Summary being processed..."

    async def _generate_key_points(self, content: str) -> List[str]:
        """Generate key points from the content"""
        try:
            prompt = f"""Extract 3-5 key points from this content.
            Return as a JSON array of strings.
            Content: {content}
            Key points:"""

            response = await self.llm.ainvoke(prompt)
            try:
                points = json.loads(response)
                if isinstance(points, list):
                    return points
            except json.JSONDecodeError:
                pass

            return ["Key point 1", "Key point 2", "Key point 3"]
        except Exception as e:
            self.logger.error(f"Error generating key points: {str(e)}")
            return ["Key points being processed..."]

    async def _generate_quiz(self, content: str) -> List[Dict]:
        """Generate quiz questions for the module"""
        try:
            prompt = f"""Create 5 multiple-choice questions based on this content.
            Return as a JSON array with this structure:
            [
                {{
                    "question": "Question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct_answer": 0,
                    "explanation": "Brief explanation of the correct answer"
                }}
            ]

            Content: {content}
            Questions:"""

            response = await self.llm.ainvoke(prompt)
            try:
                quiz = json.loads(response)
                if isinstance(quiz, list):
                    return quiz
            except json.JSONDecodeError:
                pass

            return self._generate_fallback_quiz()
        except Exception as e:
            self.logger.error(f"Error generating quiz: {str(e)}")
            return self._generate_fallback_quiz()

    def _generate_fallback_quiz(self) -> List[Dict]:
        """Generate fallback quiz if generation fails"""
        return [{
            "question": "Quiz questions are being processed. Please check back later.",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correct_answer": 0,
            "explanation": "Quiz content will be available soon."
        }]

    def create_prompt(self, content: str, style_prompt: str) -> str:
        """Create processing prompt based on content"""
        return f"""
        Transform the following educational content according to this style:
        {style_prompt}

        Guidelines:
        1. Maintain all technical accuracy and key terms
        2. Keep the educational value intact
        3. Make it engaging and memorable
        4. Add relevant examples and analogies that fit the style
        5. Break down complex concepts in the chosen style

        Content: {content}
        """

    async def update_course_content(self, course_code: str, new_module: Dict):
        """Placeholder for database update logic"""
        # This method should be implemented to update the database
        # with new modules as they're processed
        pass