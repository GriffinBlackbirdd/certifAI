# priority_processor.py
import asyncio
from typing import Dict, List, AsyncGenerator
import json
import logging

class PriorityContentProcessor:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.cache = {}  # Simple in-memory cache
        self.processing_queue = asyncio.Queue()
        self.content_status = {}
    async def process_content(self, course_code: str, pdf_content: List[str], style_prompt: str) -> AsyncGenerator[Dict, None]:
        """Process content in priority order"""
        try:
            sections = self.analyze_content_sections(pdf_content)

            priority_sections = {
                'high': ['introduction', 'key_concepts', 'core_topics'],
                'medium': ['examples', 'practice_exercises'],
                'low': ['additional_reading', 'references']
            }

            try:
                high_priority_content = await self.process_priority_section(
                    sections['high'],
                    style_prompt,
                    "Processing core content..."
                )
                yield {
                    "status": "partial",
                    "section": "core",
                    "content": high_priority_content,
                    "percentage": 30
                }
            except Exception as e:
                logging.error(f"Error processing high priority content: {str(e)}")
                yield {
                    "status": "error",
                    "section": "core",
                    "error": f"Failed to process core content: {str(e)}"
                }
                return
            try:
                medium_priority_content = await self.process_priority_section(
                    sections['medium'],
                    style_prompt,
                    "Processing examples and exercises..."
                )
                yield {
                    "status": "partial",
                    "section": "examples",
                    "content": medium_priority_content,
                    "percentage": 60
                }
            except Exception as e:
                logging.error(f"Error processing medium priority content: {str(e)}")
                medium_priority_content = []

            try:
                self.queue_background_processing(sections['low'], style_prompt)
                yield {
                    "status": "ready_for_background",
                    "message": "Core content ready. Additional content will be processed in background."
                }
            except Exception as e:
                logging.error(f"Error queueing background content: {str(e)}")

            try:
                initial_modules = self.create_initial_modules(
                    high_priority_content,
                    medium_priority_content
                )
                yield {
                    "status": "modules_ready",
                    "modules": initial_modules
                }
            except Exception as e:
                logging.error(f"Error creating initial modules: {str(e)}")
                yield {
                    "status": "error",
                    "error": f"Failed to create modules: {str(e)}"
                }

        except Exception as e:
            logging.error(f"Critical error in content processing: {str(e)}")
            yield {
                "status": "error",
                "error": f"Critical processing error: {str(e)}",
                "course_code": course_code
            }
        finally:
            self.content_status[f"{course_code}_processing"] = False
            logging.info(f"Completed processing for course: {course_code}")


        def analyze_content_sections(self, content: List[str]) -> Dict[str, List[str]]:
            """Analyze and categorize content sections by priority"""
            sections = {
                'high': [],
                'medium': [],
                'low': []
            }

            return sections
    async def process_priority_section(self, section_content: List[str], style_prompt: str, progress_message: str):
        """Process a priority section with status updates"""
        try:
            processed_content = []
            total_chunks = len(section_content)

            for i, chunk in enumerate(section_content, 1):
                try:
                    cache_key = f"{chunk[:50]}_{style_prompt[:50]}"
                    if cache_key in self.cache:
                        processed_content.append(self.cache[cache_key])
                        continue

                    response = await self.llm.ainvoke(self.create_prompt(chunk, style_prompt))
                    self.cache[cache_key] = response
                    processed_content.append(response)

                    logging.info(f"Processed chunk {i}/{total_chunks}: {progress_message}")

                except Exception as e:
                    logging.error(f"Error processing chunk {i}: {str(e)}")
                    continue

            return processed_content

        except Exception as e:
            logging.error(f"Error in priority section processing: {str(e)}")
            raise

    def queue_background_processing(self, low_priority_content: List[str], style_prompt: str):
        """Queue low-priority content for background processing"""
        try:
            for chunk in low_priority_content:
                self.processing_queue.put_nowait((chunk, style_prompt))
                logging.debug(f"Queued chunk for background processing")
        except Exception as e:
            logging.error(f"Error queueing background content: {str(e)}")
            raise

    def create_initial_modules(self, high_priority: List[str], medium_priority: List[str]) -> List[Dict]:
        """Create initial module structure with available content"""
        modules = []
        return modules

    def create_prompt(self, content: str, style_prompt: str) -> str:
        """Create processing prompt based on content priority"""
        return f"""
        Transform the following content according to style: {style_prompt}
        Content: {content}
        """

    async def get_section_status(self, course_code: str, section: str) -> Dict:
        """Get processing status of a specific section"""
        return self.content_status.get(f"{course_code}_{section}", {
            "processed": False,
            "percentage": 0
        })

    async def request_section_processing(self, course_code: str, section: str):
        """Handle on-demand processing request for a section"""
        pass