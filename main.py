from fastapi import FastAPI, Request, HTTPException, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
import uvicorn
from pathlib import Path
import sqlite3
import json
import logging
from typing import List, Dict
from datetime import datetime
import os
import asyncio
from pdf_processor import CourseProcessor
from langchain.llms import Ollama
from langchain.embeddings import FastEmbedEmbeddings
from pydantic import BaseModel
from typing import List
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="AdaptLearn")

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
pdf_dir = BASE_DIR / "pdfs"
pdf_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

llm = Ollama(model="llama3.2")
embeddings = FastEmbedEmbeddings()
course_processor = CourseProcessor(llm, embeddings)
# processor = PriorityContentProcessor(llm, embeddings)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logging.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def send_progress(self, websocket: WebSocket, data: dict):
        try:
            if websocket in self.active_connections:
                await websocket.send_json(data)
        except Exception as e:
            logging.error(f"Error sending progress: {str(e)}")
            self.disconnect(websocket)

manager = ConnectionManager()

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('courses.db')
    c = conn.cursor()

    try:
        # courses table
        c.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT UNIQUE,
                content TEXT,
                modules TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # quiz_scores table
        c.execute('''
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT,
                module_number INTEGER,
                score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses (course_code)
            )
        ''')

        conn.commit()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        conn.close()

init_database()

# Database operations
def store_course_content(course_code: str, content: str, modules: List[Dict]):
    """Store processed course content in database"""
    conn = sqlite3.connect('courses.db')
    c = conn.cursor()

    try:
        modules_json = json.dumps(modules)

        c.execute('''
            INSERT OR REPLACE INTO courses
            (course_code, content, modules, created_at)
            VALUES (?, ?, ?, datetime('now'))
        ''', (course_code, content, modules_json))

        conn.commit()
        logging.info(f"Stored course content for {course_code}")
    except Exception as e:
        logging.error(f"Error storing course content: {str(e)}")
        raise
    finally:
        conn.close()

def get_course_content(course_code: str) -> tuple:
    """Retrieve course content from database"""
    conn = sqlite3.connect('courses.db')
    c = conn.cursor()

    try:
        c.execute('''
            SELECT content, modules
            FROM courses
            WHERE course_code = ?
        ''', (course_code,))

        result = c.fetchone()
        if result:
            content, modules_json = result
            modules = json.loads(modules_json) if modules_json else []
            return content, modules
        return None, None
    except Exception as e:
        logging.error(f"Error retrieving course content: {str(e)}")
        raise
    finally:
        conn.close()

# Routes
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/courses", response_class=HTMLResponse)
async def courses_page(request: Request):
    return templates.TemplateResponse("courses.html", {"request": request})

@app.get("/course/{course_code}/customize", response_class=HTMLResponse)
async def customize_learning(request: Request, course_code: str):
    if course_code.lower() not in ["dp900", "ai900", "az900"]:
        raise HTTPException(status_code=404, detail="Course not found")
    return templates.TemplateResponse(
        "style-customization.html",
        {"request": request, "course_code": course_code}
    )

@app.get("/course/{course_code}/loading", response_class=HTMLResponse)
async def loading_page(request: Request, course_code: str, style: str):
    return templates.TemplateResponse(
        "loading.html",
        {
            "request": request,
            "course_code": course_code,
            "style": style
        }
    )
@app.websocket("/ws/course/{course_code}")
async def course_processing_websocket(websocket: WebSocket, course_code: str):
    await manager.connect(websocket)
    try:
        data = await websocket.receive_text()
        if data.startswith("start_processing:"):
            style_prompt = data.split(":", 1)[1]
            pdf_path = pdf_dir / f"{course_code}.pdf"

            async for progress in course_processor.process_pdf(str(pdf_path), style_prompt):
                if websocket not in manager.active_connections:
                    break

                logging.info(f"Processing progress: {progress}")

                if progress["status"] == "partial":
                    try:
                        # Store first module
                        store_course_content(
                            course_code=course_code,
                            content=progress["content"],
                            modules=progress["modules"]
                        )
                        logging.info(f"Stored first module for {course_code}")

                        # Send redirect immediately after first module is ready
                        await manager.send_progress(websocket, {
                            "status": "redirect",
                            "url": f"/course/{course_code}/learn"
                        })

                    except Exception as e:
                        logging.error(f"Failed to store content: {str(e)}")
                        await manager.send_progress(websocket, {
                            "status": "error",
                            "error": "Failed to save course content"
                        })
                        break

                elif progress["status"] == "processing":
                    # Send processing updates
                    await manager.send_progress(websocket, progress)

                elif progress["status"] == "completed":
                    try:
                        # Store all modules
                        store_course_content(
                            course_code=course_code,
                            content=progress["content"],
                            modules=progress["modules"]
                        )
                        logging.info(f"Stored all modules for {course_code}")

                    except Exception as e:
                        logging.error(f"Error storing final content: {str(e)}")
                        await manager.send_progress(websocket, {
                            "status": "error",
                            "error": "Failed to save complete course content"
                        })

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for {course_code}")
    except Exception as e:
        logging.error(f"WebSocket error for {course_code}: {str(e)}")
        if websocket in manager.active_connections:
            try:
                await manager.send_progress(websocket, {
                    "status": "error",
                    "error": str(e)
                })
            except:
                pass
    finally:
        manager.disconnect(websocket)
@app.get("/course/{course_code}/processing-status")
async def get_processing_status(course_code: str):
    """Get the current processing status for a course"""
    try:
        statuses = {
            "core": await processor.get_section_status(course_code, "core"),
            "examples": await processor.get_section_status(course_code, "examples"),
            "additional": await processor.get_section_status(course_code, "additional")
        }
        return {
            "course_code": course_code,
            "statuses": statuses,
            "overall_status": "processing" if any(not s["processed"] for s in statuses.values()) else "completed"
        }
    except Exception as e:
        logging.error(f"Error getting processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/course/{course_code}/status")
async def check_processing_status(course_code: str):
    """Check the status of background processing"""
    status = await course_processor.get_processing_status(course_code)
    return status

@app.post("/course/{course_code}/process-section/{section}")
async def request_section_processing(course_code: str, section: str):
    """Request processing for a specific section"""
    try:
        await processor.request_section_processing(course_code, section)
        return {
            "status": "processing_started",
            "course_code": course_code,
            "section": section
        }
    except Exception as e:
        logging.error(f"Error requesting section processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/course/{course_code}/learn", response_class=HTMLResponse)
async def learning_page(request: Request, course_code: str):
    """Get current course content"""
    content, modules = get_course_content(course_code)

    # if only first module is ready i am showing it
    if content and modules:
        return templates.TemplateResponse(
            "learning_page.html",
            {
                "request": request,
                "course_code": course_code,
                "content": content,
                "modules": modules,
                "is_processing": await course_processor.get_processing_status(course_code)
            }
        )

    raise HTTPException(status_code=404, detail="Course content not found")

class QuizSubmission(BaseModel):
    answers: List[int]

@app.post("/course/{course_code}/quiz/{module_number}")
async def submit_quiz(
    course_code: str,
    module_number: int,
    submission: QuizSubmission
):
    """Handle quiz submission"""
    try:
        logging.info(f"Received quiz submission for {course_code}, module {module_number}")
        logging.info(f"Answers received: {submission.answers}")

        content, modules = get_course_content(course_code)
        if not modules:
            raise HTTPException(status_code=404, detail="Course content not found")

        if isinstance(modules, str):
            modules = json.loads(modules)

        module = next((m for m in modules if m["number"] == module_number), None)
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")

        correct_answers = [q["correct_answer"] for q in module["quiz"]]
        if len(submission.answers) != len(correct_answers):
            raise HTTPException(
                status_code=400,
                detail="Number of answers doesn't match number of questions"
            )

        score = sum(
            1 for a, c in zip(submission.answers, correct_answers) if a == c
        ) / len(correct_answers) * 100

        conn = sqlite3.connect('courses.db')
        try:
            c = conn.cursor()
            c.execute('''
                INSERT INTO quiz_scores (course_code, module_number, score)
                VALUES (?, ?, ?)
            ''', (course_code, module_number, score))
            conn.commit()
        finally:
            conn.close()

        return {
            "score": score,
            "total_questions": len(correct_answers),
            "correct_answers": sum(1 for a, c in zip(submission.answers, correct_answers) if a == c),
            "message": "Great job!" if score >= 70 else "Keep practicing!"
        }

    except Exception as e:
        logging.error(f"Error processing quiz submission: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/course/{course_code}/quiz/{module_number}/status")
async def check_quiz_status(course_code: str, module_number: int):
    try:
        content, modules = get_course_content(course_code)
        if not modules:
            return {"status": "not_ready"}

        if isinstance(modules, str):
            modules = json.loads(modules)

        module = next((m for m in modules if m["number"] == module_number), None)
        if not module or not module.get("quiz"):
            return {"status": "not_ready"}

        return {"status": "ready"}
    except Exception as e:
        logging.error(f"Error checking quiz status: {str(e)}")
        return {"status": "error"}


@app.get("/course/{course_code}/modules/status")
async def check_modules_status(course_code: str):
    """Check the status of module processing"""
    try:
        logging.info(f"Checking module status for {course_code}")
        content, modules = get_course_content(course_code)

        if not modules:
            logging.info("No modules found - Processing not started")
            return {
                "status": "processing",
                "percentage": 0,
                "current_module": 2,
                "current_step": 0,
                "message": "Starting module processing..."
            }

        if isinstance(modules, str):
            modules = json.loads(modules)

        logging.info(f"Found {len(modules)} modules")

        if len(modules) > 1:
            logging.info("Additional modules are ready")
            additional_modules = modules[1:]  #
            logging.info(f"Sending {len(additional_modules)} additional modules")
            return {
                "status": "completed",
                "modules": additional_modules
            }

        logging.info("Still processing additional modules")
        return {
            "status": "processing",
            "percentage": 30,
            "current_module": 2,
            "current_step": 1,
            "message": "Transforming content..."
        }

    except Exception as e:
        logging.error(f"Error checking modules status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Error checking module status"
        }


def get_module_processing_status(course_code: str) -> dict:
    """Get the current processing status for modules"""
    try:
        content, modules = get_course_content(course_code)
        if not modules:
            return {
                "is_processing": True,
                "modules_count": 0,
                "percentage": 0
            }

        if isinstance(modules, str):
            modules = json.loads(modules)

        return {
            "is_processing": len(modules) <= 1,
            "modules_count": len(modules),
            "percentage": min((len(modules) / 4) * 100, 100)  # Assuming 4 total modules
        }
    except Exception as e:
        logging.error(f"Error getting module status: {str(e)}")
        return {
            "is_processing": True,
            "modules_count": 0,
            "percentage": 0
        }

async def monitor_module_processing(course_code: str):
    """Background task to monitor module processing"""
    while True:
        try:
            status = get_module_processing_status(course_code)
            if not status["is_processing"]:
                logging.info(f"Module processing completed for {course_code}")
                break

            logging.info(f"Module processing status for {course_code}: {status}")
            await asyncio.sleep(5)  #

        except Exception as e:
            logging.error(f"Error monitoring modules for {course_code}: {str(e)}")
            await asyncio.sleep(5)

@app.websocket("/ws/course/{course_code}")
async def course_processing_websocket(websocket: WebSocket, course_code: str):
    await manager.connect(websocket)
    try:
        data = await websocket.receive_text()
        if data.startswith("start_processing:"):
            style_prompt = data.split(":", 1)[1]
            pdf_path = pdf_dir / f"{course_code}.pdf"

            # Start monitoring in background
            monitoring_task = asyncio.create_task(monitor_module_processing(course_code))

            async for progress in course_processor.process_pdf(str(pdf_path), style_prompt):
                if websocket not in manager.active_connections:
                    break

                logging.info(f"Processing progress: {progress}")

                if progress["status"] == "partial":
                    try:
                        store_course_content(
                            course_code=course_code,
                            content=progress["content"],
                            modules=progress["modules"]
                        )
                        logging.info(f"Stored partial content for {course_code}")

                        await manager.send_progress(websocket, {
                            "status": "progress",
                            "message": "First module ready, processing additional content...",
                            "percentage": 30
                        })

                    except Exception as e:
                        logging.error(f"Failed to store content: {str(e)}")
                        await manager.send_progress(websocket, {
                            "status": "error",
                            "error": "Failed to save course content"
                        })
                        break

                elif progress["status"] == "completed":
                    try:
                        store_course_content(
                            course_code=course_code,
                            content=progress["content"],
                            modules=progress["modules"]
                        )
                        logging.info(f"Stored complete content for {course_code}")

                        await manager.send_progress(websocket, {
                            "status": "completed",
                            "message": "All modules ready",
                            "redirect": f"/course/{course_code}/learn"
                        })

                    except Exception as e:
                        logging.error(f"Error storing final content: {str(e)}")
                        await manager.send_progress(websocket, {
                            "status": "error",
                            "error": "Failed to save complete course content"
                        })

                try:
                    await manager.send_progress(websocket, progress)
                except Exception as e:
                    logging.error(f"Error sending progress: {str(e)}")
                    break

            monitoring_task.cancel()

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for {course_code}")
    except Exception as e:
        logging.error(f"WebSocket error for {course_code}: {str(e)}")
        if websocket in manager.active_connections:
            try:
                await manager.send_progress(websocket, {
                    "status": "error",
                    "error": str(e)
                })
            except:
                pass
    finally:
        manager.disconnect(websocket)

def calculate_exam_readiness(scores: List[float], weights: List[float] = None) -> dict:
    """
    Calculate exam readiness using Bayesian approach
    """
    if not scores:
        return {
            "ready_date": None,
            "confidence": 0,
            "recommendation": "Complete more quizzes to get a prediction"
        }

    # some prior beliefs
    prior_mean = 70  # Expected passing score
    prior_std = 10   # Uncertainty
    # if no weights provided i will use recency weights
    if weights is None:
        weights = [1 + i/len(scores) for i in range(len(scores))]
    # weighted mean and std of scores
    weighted_scores = np.array(scores) * np.array(weights)
    mean_score = np.average(weighted_scores)
    std_score = np.std(scores) if len(scores) > 1 else 15
    # bayesian update
    posterior_var = 1 / (1/prior_std**2 + 1/std_score**2)
    posterior_mean = posterior_var * (prior_mean/prior_std**2 + mean_score/std_score**2)
    # confidence level (0-100)
    confidence = min(100, (mean_score/posterior_mean * 100))
    # days needed based on current performance
    score_gap = max(0, prior_mean - posterior_mean)
    base_days = 30  # Base preparation time
    additional_days = int(score_gap * 0.7)  # 0.7 days per point needed
    total_days = base_days + additional_days

    # ready date
    ready_date = datetime.now() + timedelta(days=total_days)
    if confidence > 80:
        recommendation = "You're on track! Keep up the good work."
    elif confidence > 60:
        recommendation = "Making good progress. Focus on topics where you scored lower."
    else:
        recommendation = "More practice recommended. Try reviewing the modules again."

    return {
        "ready_date": ready_date.strftime("%Y-%m-%d"),
        "confidence": round(confidence, 1),
        "recommendation": recommendation,
        "days_needed": total_days
    }

@app.get("/course/{course_code}/progress")
async def get_course_progress(course_code: str):
    """Get course progress and exam prediction"""
    try:
        conn = sqlite3.connect('courses.db')
        c = conn.cursor()
        c.execute('''
            SELECT score, timestamp
            FROM quiz_scores
            WHERE course_code = ?
            ORDER BY timestamp DESC
        ''', (course_code,))

        results = c.fetchall()
        conn.close()

        if not results:
            return {
                "status": "no_data",
                "message": "No quiz data available yet"
            }

        scores = [row[0] for row in results]
        prediction = calculate_exam_readiness(scores)

        return {
            "status": "success",
            "average_score": sum(scores) / len(scores),
            "completion_percentage": (len(scores) / 4) * 100,  # Assuming 4 modules
            "prediction": prediction
        }

    except Exception as e:
        logging.error(f"Error getting course progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)