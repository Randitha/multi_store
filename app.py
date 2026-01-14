import os
import json
import glob
import time
import hashlib
import logging
import threading
import uuid
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, session, send_from_directory, g
from flask_cors import CORS
from flask_compress import Compress
from flask_session import Session
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from datetime import timedelta, datetime
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
Compress(app)

# Advanced cache configuration
DATA_CACHE = {}
CACHE_TIMEOUT = 300  # 5 minutes cache timeout
MAX_DATA_SIZE = 5000  # Minimal size to avoid API limits
SUGGESTION_SAMPLE_SIZE = 10  # Very small for fast suggestions
MAX_PARALLEL_CHUNKS = 4  # Maximum chunks to process in parallel
CHUNK_PROCESSING_TIMEOUT = 30  # Timeout for chunk processing
INTELLIGENT_CHUNK_SELECTION = True  # Enable smart chunk selection

# Setup advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_performance.log')
    ]
)

# Create custom logger with request ID support
class RequestFormatter(logging.Formatter):
    def format(self, record):
        try:
            # Try to get request ID from Flask's g object
            from flask import has_request_context, g
            if has_request_context():
                record.request_id = getattr(g, 'request_id', 'NO_ID')
            else:
                record.request_id = 'STARTUP'
        except (RuntimeError, ImportError):
            record.request_id = 'SYSTEM'
        return super().format(record)

for handler in logging.getLogger().handlers:
    handler.setFormatter(RequestFormatter(
        '%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s'
    ))

app.logger.setLevel(logging.INFO)

# Performance metrics storage
PERFORMANCE_METRICS = {
    'total_requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'average_response_time': 0,
    'parallel_processing_count': 0
}

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS, thread_name_prefix="ChunkProcessor")

# Enable CORS with credentials support
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Configure Flask-Session with 1-hour expiration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # 1 hour expiration
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_MAX_AGE'] = 3600  # 1 hour in seconds
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_FILE_THRESHOLD'] = 100  # Cleanup when more than 100 session files

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)

# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def cleanup_expired_sessions():
    """
    Clean up expired session files from the filesystem
    """
    try:
        session_dir = app.config['SESSION_FILE_DIR']
        current_time = time.time()
        expiry_time = 3600  # 1 hour in seconds

        session_files = glob.glob(os.path.join(session_dir, 'session_*'))
        for session_file in session_files:
            try:
                # Check file modification time
                file_mtime = os.path.getmtime(session_file)
                if current_time - file_mtime > expiry_time:
                    os.remove(session_file)
            except (OSError, Exception):
                # If we can't delete a file, continue with others
                continue
    except Exception:
        # If cleanup fails, don't break the application
        pass

# Cache management functions
def get_cache_key(params=None):
    """Generate a cache key based on parameters"""
    if params is None:
        params = {}
    
    # Create a hash of parameters for cache key
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def get_cached_data(cache_key):
    """Get cached data if still valid"""
    if cache_key in DATA_CACHE:
        cached_data, timestamp = DATA_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TIMEOUT:
            PERFORMANCE_METRICS['cache_hits'] += 1
            app.logger.info(f"Cache HIT for key: {cache_key[:16]}...")
            return cached_data
        else:
            # Remove expired cache entry
            del DATA_CACHE[cache_key]
            app.logger.info(f"Cache entry expired and removed: {cache_key[:16]}...")
    
    PERFORMANCE_METRICS['cache_misses'] += 1
    app.logger.info(f"Cache MISS for key: {cache_key[:16]}...")
    return None

def set_cached_data(cache_key, data):
    """Set data in cache with timestamp"""
    DATA_CACHE[cache_key] = (data, time.time())
    data_size = len(data.get('data', [])) if isinstance(data, dict) else 0
    app.logger.info(f"Data cached: {data_size} records for key: {cache_key[:16]}...")
    
    # Clean up cache if it gets too large (keep only 10 most recent)
    if len(DATA_CACHE) > 10:
        oldest_key = min(DATA_CACHE.keys(), key=lambda k: DATA_CACHE[k][1])
        del DATA_CACHE[oldest_key]
        app.logger.info(f"Cache cleanup: Removed oldest entry {oldest_key[:16]}...")

def clear_cache():
    """Clear all cached data"""
    global DATA_CACHE
    DATA_CACHE = {}

# Import functions from single_store
from single_store import (
    query_tinybird_api,
    analyze_data_with_gemini_stream,
    analyze_data_with_parallel_processing,
    generate_chart_with_gemini,
    model,
    clean_gemini_response,
    has_zero_results,
    get_sample_data_for_suggestions,
    query_tinybird_with_limit,
    intelligent_chunk_selection
)

@app.before_request
def before_request():
    """
    Run before each request - setup logging and cleanup
    """
    # Generate unique request ID for tracking
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = time.time()
    
    # Log request start
    app.logger.info(f"Request started: {request.method} {request.path} from {request.remote_addr}")
    
    # Update metrics
    PERFORMANCE_METRICS['total_requests'] += 1
    
    # Clean up expired sessions periodically (every 10th request)
    if hasattr(app, 'request_count'):
        app.request_count += 1
    else:
        app.request_count = 1

    if app.request_count % 10 == 0:
        cleanup_expired_sessions()
        app.logger.info("Periodic session cleanup completed")

@app.after_request
def after_request(response):
    """
    Run after each request - log performance metrics
    """
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        
        # Update average response time
        current_avg = PERFORMANCE_METRICS['average_response_time']
        total_requests = PERFORMANCE_METRICS['total_requests']
        PERFORMANCE_METRICS['average_response_time'] = (
            (current_avg * (total_requests - 1) + duration) / total_requests
        )
        
        # Log request completion
        app.logger.info(
            f"Request completed: {response.status_code} in {duration:.3f}s - "
            f"Cache hits: {PERFORMANCE_METRICS['cache_hits']}, "
            f"Parallel processing: {PERFORMANCE_METRICS['parallel_processing_count']}"
        )
    
    return response

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('ask_question')
def handle_ask_question(data):
    """
    Handle incoming question with parallel processing and stream response back
    """
    request_start = time.time()
    question_id = str(uuid.uuid4())[:8]
    
    try:
        question = data.get('question', '').strip()
        if not question:
            app.logger.warning(f"[{question_id}] Empty question received")
            emit('error', {'error': 'No question provided'})
            return
        
        app.logger.info(f"[{question_id}] Processing question: {question[:50]}...")
        
        # Use cached data or fetch with limit for better performance
        cache_key = get_cache_key({'limit': MAX_DATA_SIZE})
        tinybird_data = get_cached_data(cache_key)
        
        if tinybird_data is None:
            app.logger.info(f"[{question_id}] Fetching fresh data from Tinybird")
            data_start = time.time()
            tinybird_data = query_tinybird_with_limit(MAX_DATA_SIZE)
            data_time = time.time() - data_start
            app.logger.info(f"[{question_id}] Data fetched in {data_time:.2f}s")
            
            if 'error' not in tinybird_data:
                set_cached_data(cache_key, tinybird_data)
        else:
            app.logger.info(f"[{question_id}] Using cached data")
        
        if 'error' in tinybird_data:
            app.logger.error(f"[{question_id}] Tinybird error: {tinybird_data['error']}")
            emit('error', {'error': tinybird_data['error']})
            return

        # Check if this question would have zero results
        if has_zero_results(question, tinybird_data):
            # Add debugging information
            data_size = len(tinybird_data.get('data', [])) if isinstance(tinybird_data, dict) else 0
            app.logger.warning(f"[{question_id}] Question flagged as zero results - Data size: {data_size}")
            
            response_data = {
                "question": question,
                "answer": "I don't have enough data to answer this question based on the current dataset.",
                "has_chart_potential": False,
                "data_source": "Tinybird Pipe: main_pipe (Optimized)",
                "zero_results": True,
                "debug_info": f"Data records available: {data_size}"
            }
            emit('complete_response', response_data)
            
            # Update session history
            if 'store_history' not in session:
                session['store_history'] = []
            
            session['store_history'].append({'type': 'user', 'message': question})
            session['store_history'].append({
                'type': 'assistant',
                'message': response_data['answer'],
                'has_chart_potential': False,
                'zero_results': True
            })
            
            if len(session['store_history']) > 20:
                session['store_history'] = session['store_history'][-20:]
            
            session.modified = True
            return

        # Use parallel processing for faster analysis
        analysis_start = time.time()
        app.logger.info(f"[{question_id}] Starting parallel data analysis")
        PERFORMANCE_METRICS['parallel_processing_count'] += 1
        
        full_answer = ""
        # Try parallel processing first, fallback to streaming if needed
        try:
            for chunk in analyze_data_with_parallel_processing(question, tinybird_data, question_id):
                full_answer += chunk
                # Emit each chunk as it's generated
                emit('stream_chunk', {'chunk': chunk})
        except Exception as parallel_error:
            app.logger.warning(f"[{question_id}] Parallel processing failed, falling back to streaming: {parallel_error}")
            # Fallback to original streaming method
            for chunk in analyze_data_with_gemini_stream(question, tinybird_data):
                full_answer += chunk
                emit('stream_chunk', {'chunk': chunk})
        
        analysis_time = time.time() - analysis_start
        app.logger.info(f"[{question_id}] Analysis completed in {analysis_time:.2f}s")

        # Emit completion signal
        emit('stream_complete', {'complete': True})

        # Check if this response might benefit from a chart
        chart_related_keywords = ['trend', 'comparison', 'over time', 'percentage', 'distribution',
                                 'growth', 'change', 'increase', 'decrease', 'ratio', 'proportion',
                                 'average', 'total', 'count', 'peak', 'highest', 'lowest']

        question_lower = question.lower()
        answer_lower = full_answer.lower()  # full_answer is already defined above
        
        # Check both question and answer for chart potential
        has_chart_potential = (
            any(keyword in question_lower for keyword in chart_related_keywords) or
            any(keyword in answer_lower for keyword in chart_related_keywords)
        )

        # Initialize session chat history if not exists
        if 'store_history' not in session:
            session['store_history'] = []

        # Add to session-based chat history (keep only last 20 messages)
        session['store_history'].append({
            'type': 'user',
            'message': question
        })

        session['store_history'].append({
            'type': 'assistant',
            'message': full_answer,
            'has_chart_potential': has_chart_potential,
            'zero_results': False
        })

        # Keep only last 20 messages
        if len(session['store_history']) > 20:
            session['store_history'] = session['store_history'][-20:]

        # Mark session as modified to update expiration
        session.modified = True

        # Send final complete response data with performance info
        total_time = time.time() - request_start
        response_data = {
            "question": question,
            "answer": full_answer,
            "has_chart_potential": has_chart_potential,
            "data_source": "Tinybird Pipe: main_pipe (Parallel Optimized)",
            "zero_results": False,
            "processing_time": f"{total_time:.2f}s",
            "request_id": question_id
        }
        emit('complete_response', response_data)
        
        app.logger.info(
            f"[{question_id}] Question processed successfully in {total_time:.2f}s - "
            f"Answer length: {len(full_answer)} chars"
        )

    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"[{question_id}] Error in WebSocket handler: {error_msg}")
        emit('error', {'error': f"An unexpected error occurred: {error_msg}"})
@app.route('/store/ask', methods=['POST'])
def store_ask_question():
    """
    Optimized endpoint for non-WebSocket clients with parallel processing
    """
    request_start = time.time()
    question_id = str(uuid.uuid4())[:8]
    
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            app.logger.warning(f"[{question_id}] No question provided in request")
            return jsonify({"error": "No question provided"}), 400

        question = data['question'].strip()
        if not question:
            app.logger.warning(f"[{question_id}] Empty question provided")
            return jsonify({"error": "Empty question provided"}), 400

        app.logger.info(f"[{question_id}] REST API processing question: {question[:50]}...")

        # Use cached data or fetch with limit for better performance
        cache_key = get_cache_key({'limit': MAX_DATA_SIZE})
        tinybird_data = get_cached_data(cache_key)
        
        if tinybird_data is None:
            app.logger.info(f"[{question_id}] Fetching fresh data from Tinybird")
            data_start = time.time()
            tinybird_data = query_tinybird_with_limit(MAX_DATA_SIZE)
            data_time = time.time() - data_start
            app.logger.info(f"[{question_id}] Data fetched in {data_time:.2f}s")
            
            if 'error' not in tinybird_data:
                set_cached_data(cache_key, tinybird_data)
        else:
            app.logger.info(f"[{question_id}] Using cached data")
        
        if 'error' in tinybird_data:
            app.logger.error(f"[{question_id}] Tinybird error: {tinybird_data['error']}")
            return jsonify({"error": tinybird_data['error']}), 500

        # Temporarily disable zero results check to test data flow
        # if has_zero_results(question, tinybird_data):
        #     # Add debugging information
        #     data_size = len(tinybird_data.get('data', [])) if isinstance(tinybird_data, dict) and 'data' in tinybird_data else 0
        #     app.logger.warning(f"[{question_id}] REST - Question flagged as zero results - Data size: {data_size}")
        #     
        #     return jsonify({
        #         "question": question,
        #         "answer": "I don't have enough data to answer this question based on the current dataset.",
        #         "has_chart_potential": False,
        #         "data_source": "Tinybird Pipe: main_pipe (Optimized)",
        #         "zero_results": True,
        #         "processing_time": f"{time.time() - request_start:.2f}s",
        #         "request_id": question_id,
        #         "debug_info": f"Data records available: {data_size}"
        #     })

        # Use parallel processing for analysis - collect full response
        analysis_start = time.time()
        app.logger.info(f"[{question_id}] Starting parallel data analysis for REST API")
        PERFORMANCE_METRICS['parallel_processing_count'] += 1
        
        answer = ""
        try:
            for chunk in analyze_data_with_parallel_processing(question, tinybird_data, question_id):
                answer += chunk
        except Exception as parallel_error:
            app.logger.warning(f"[{question_id}] Parallel processing failed, falling back to streaming: {parallel_error}")
            # Fallback to original streaming method
            for chunk in analyze_data_with_gemini_stream(question, tinybird_data):
                answer += chunk
        
        analysis_time = time.time() - analysis_start
        app.logger.info(f"[{question_id}] Analysis completed in {analysis_time:.2f}s")

        # Check if this response might benefit from a chart
        chart_related_keywords = ['trend', 'comparison', 'over time', 'percentage', 'distribution',
                                 'growth', 'change', 'increase', 'decrease', 'ratio', 'proportion',
                                 'average', 'total', 'count', 'peak', 'highest', 'lowest']

        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Check both question and answer for chart potential
        has_chart_potential = (
            any(keyword in question_lower for keyword in chart_related_keywords) or
            any(keyword in answer_lower for keyword in chart_related_keywords)
        )

        # Initialize session chat history if not exists
        if 'store_history' not in session:
            session['store_history'] = []

        # Add to session-based chat history (keep only last 20 messages)
        session['store_history'].append({
            'type': 'user',
            'message': question
        })

        session['store_history'].append({
            'type': 'assistant',
            'message': answer,
            'has_chart_potential': has_chart_potential,
            'zero_results': False
        })

        # Keep only last 20 messages
        if len(session['store_history']) > 20:
            session['store_history'] = session['store_history'][-20:]

        # Mark session as modified to update expiration
        session.modified = True

        total_time = time.time() - request_start
        app.logger.info(
            f"[{question_id}] REST API question processed successfully in {total_time:.2f}s - "
            f"Answer length: {len(answer)} chars"
        )

        return jsonify({
            "question": question,
            "answer": answer,
            "has_chart_potential": has_chart_potential,
            "data_source": "Tinybird Pipe: main_pipe (Parallel Optimized)",
            "zero_results": False,
            "processing_time": f"{total_time:.2f}s",
            "request_id": question_id
        })

    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"[{question_id}] Error in REST API handler: {error_msg}")
        return jsonify({"error": f"An unexpected error occurred: {error_msg}"}), 500
@app.route('/store/generate-chart', methods=['POST'])
def store_generate_chart():
    """
    Endpoint to generate a chart for a specific question and answer
    """
    try:
        # Get question and answer from request
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({"error": "Question and answer required"}), 400

        question = data['question'].strip()
        answer = data['answer'].strip()

        # Check if this is a zero-results scenario
        if "don't have enough data" in answer.lower() or "no data" in answer.lower():
            return jsonify({
                "chart_html": "<div class='chart-error'>No data available to generate a chart for this question.</div>",
                "question": question,
                "zero_results": True
            })

        # Use cached data or fetch with limit for better performance
        cache_key = get_cache_key({'limit': MAX_DATA_SIZE})
        tinybird_data = get_cached_data(cache_key)
        
        if tinybird_data is None:
            # Fetch limited data and cache it
            tinybird_data = query_tinybird_with_limit(MAX_DATA_SIZE)
            if 'error' not in tinybird_data:
                set_cached_data(cache_key, tinybird_data)
        
        if 'error' in tinybird_data:
            return jsonify({"error": tinybird_data['error']}), 500

        # Check if this question would have zero results
        if has_zero_results(question, tinybird_data):
            return jsonify({
                "chart_html": "<div class='chart-error'>No data available to generate a chart for this question.</div>",
                "question": question,
                "zero_results": True
            })

        # Use Gemini to generate a chart
        chart_html = generate_chart_with_gemini(question, tinybird_data, answer)

        # Update the session history with the chart HTML
        if 'store_history' in session:
            for i, msg in enumerate(session['store_history']):
                if msg['type'] == 'assistant' and msg['message'] == answer:
                    session['store_history'][i]['chart_html'] = chart_html
                    break

            session.modified = True

        return jsonify({
            "chart_html": chart_html,
            "question": question,
            "zero_results": False
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate chart: {str(e)}"}), 500

def generate_ai_suggestions(chat_history=None, sample_data=None):
    """Generate AI-powered suggestions using Gemini based on chat history and data context"""
    try:
        suggestion_start = time.time()
        
        # Create cache key based on recent questions (last 3 user questions)
        cache_parts = []
        if chat_history and len(chat_history) > 0:
            for item in chat_history[-5:]:
                if item.get('type') == 'user':
                    cache_parts.append(item.get('message', '')[:50])  # First 50 chars of each question
        
        cache_key = f"suggestions_{hashlib.md5('_'.join(cache_parts).encode()).hexdigest()[:12]}" if cache_parts else "suggestions_initial"
        
        # Check cache first (2 minute cache for suggestions to allow fresh contextual updates)
        if cache_key in DATA_CACHE:
            cached_data, cached_time = DATA_CACHE[cache_key]
            if time.time() - cached_time < 120:  # 2 minutes
                app.logger.info(f"Using cached AI suggestions - Age: {time.time() - cached_time:.1f}s")
                return cached_data
        
        # Create the model for suggestions - use faster model
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',  # Using same model as main chat
            generation_config={
                'temperature': 0.9,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 400,
            }
        )
        
        # Build the prompt based on context
        if chat_history and len(chat_history) > 0:
            # Get recent user questions
            recent_questions = []
            for item in chat_history[-5:]:
                if item.get('type') == 'user':
                    recent_questions.append(item.get('message', ''))
            
            if recent_questions:
                app.logger.info(f"Generating contextual suggestions based on {len(recent_questions)} recent questions")
                prompt = f"""You are an AI assistant helping with retail store visitor analytics. 

Recent conversation questions:
{chr(10).join(f'- {q}' for q in recent_questions)}

Based on this conversation context, generate 8 follow-up questions that would provide deeper insights or explore related aspects of the data. The questions should:
1. Build upon topics already discussed
2. Explore related analytical angles
3. Suggest comparisons or trends
4. Be specific and actionable

Return ONLY the 8 questions, one per line, without numbering or bullets."""
            else:
                recent_questions = None
        else:
            recent_questions = None
        
        # If no contextual questions, generate initial suggestions
        if not recent_questions:
            # Initial suggestions - based on available data
            app.logger.info("Generating initial AI suggestions (no context)")
            data_context = ""
            if sample_data:
                data_context = f"\nSample data fields available: {', '.join(sample_data.keys()) if isinstance(sample_data, dict) else 'visitor analytics data'}"
            
            prompt = f"""You are an AI assistant helping with retail store visitor analytics.{data_context}

Generate 8 insightful questions that would be valuable for analyzing retail store visitor data. Focus on:
1. Demographic patterns (age, gender)
2. Time-based trends (hourly, daily, weekly)
3. Visitor behavior and patterns
4. Comparative analysis
5. Peak times and traffic flow
6. Growth and changes over time

Return ONLY the 8 questions, one per line, without numbering or bullets."""
        
        # Generate suggestions using Gemini
        response = model.generate_content(prompt)
        suggestions_text = response.text.strip()
        
        # Parse the suggestions
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            # Remove numbering, bullets, or dashes
            line = line.lstrip('0123456789.-•* ')
            if line and len(line) > 10:  # Valid question
                suggestions.append(line)
        
        # Cache the results
        DATA_CACHE[cache_key] = (suggestions[:8], time.time())
        
        generation_time = time.time() - suggestion_start
        app.logger.info(f"AI suggestions generated in {generation_time:.3f}s - Count: {len(suggestions)}")
        
        return suggestions[:8]  # Return exactly 8 suggestions
        
    except Exception as e:
        app.logger.error(f"Error generating AI suggestions: {e}")
        # Return None to indicate failure - caller should handle
        return None
@app.route('/store/suggestions', methods=['GET'])
def store_get_suggestions():
    """
    AI-powered endpoint to generate contextual suggestions
    """
    request_start = time.time()
    suggestion_id = str(uuid.uuid4())[:8]
    
    try:
        app.logger.info(f"[{suggestion_id}] Generating AI-powered suggestions")
        
        # Get chat history to determine context
        chat_history = session.get('store_history', [])
        
        # Get sample data for initial context (only if no chat history)
        sample_data = None
        if len(chat_history) == 0:
            try:
                from single_store import get_sample_data_for_suggestions
                sample_data = get_sample_data_for_suggestions()
            except Exception as e:
                app.logger.warning(f"[{suggestion_id}] Could not get sample data: {e}")
        
        # Generate AI-powered suggestions
        suggestions = generate_ai_suggestions(chat_history, sample_data)
        
        if not suggestions or len(suggestions) == 0:
            app.logger.error(f"[{suggestion_id}] AI suggestion generation failed")
            return jsonify({
                "error": "Failed to generate suggestions",
                "request_id": suggestion_id
            }), 500
        
        total_time = time.time() - request_start
        app.logger.info(f"[{suggestion_id}] AI suggestions generated in {total_time:.3f}s - Count: {len(suggestions)}")
        
        return jsonify({
            "suggested_questions": suggestions,
            "generation_time": f"{total_time:.3f}s",
            "request_id": suggestion_id,
            "contextual": len(chat_history) > 0,
            "ai_generated": True
        })

    except Exception as e:
        app.logger.error(f"[{suggestion_id}] Error in suggestions endpoint: {str(e)}")
        return jsonify({
            "error": f"Failed to generate suggestions: {str(e)}",
            "request_id": suggestion_id
        }), 500

@app.route('/store/chat-history', methods=['GET'])
def store_get_chat_history():
    """
    Endpoint to get store history
    """
    try:
        # Return session-based chat history
        return jsonify({
            "store_history": session.get('store_history', [])
        })

    except Exception as e:
        return jsonify({"error": "Failed to get store history"}), 500

@app.route('/store/clear-history', methods=['POST'])
def store_clear_chat_history():
    """
    Endpoint to clear store history
    """
    try:
        # Clear the session-based chat history
        session['store_history'] = []
        session.modified = True

        return jsonify({
            "status": "success",
            "message": "Store history cleared"
        })

    except Exception as e:
        return jsonify({"error": "Failed to clear store history"}), 500
@app.route('/cleanup-sessions', methods=['POST'])
def manual_cleanup_sessions():
    """
    Manual endpoint to clean up expired sessions
    """
    try:
        cleanup_expired_sessions()
        return jsonify({
            "status": "success",
            "message": "Session cleanup completed"
        })
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_data_cache():
    """
    Manual endpoint to clear data cache
    """
    try:
        clear_cache()
        return jsonify({
            "status": "success",
            "message": "Data cache cleared",
            "cache_cleared": True
        })
    except Exception as e:
        return jsonify({"error": f"Cache clear failed: {str(e)}"}), 500

@app.route('/cache-status', methods=['GET'])
def get_cache_status():
    """
    Get current cache status
    """
    try:
        cache_info = {
            "cache_entries": len(DATA_CACHE),
            "cache_timeout": CACHE_TIMEOUT,
            "max_data_size": MAX_DATA_SIZE,
            "suggestion_sample_size": SUGGESTION_SAMPLE_SIZE,
            "cached_keys": list(DATA_CACHE.keys()),
            "parallel_processing": {
                "max_workers": MAX_PARALLEL_CHUNKS,
                "timeout": CHUNK_PROCESSING_TIMEOUT,
                "intelligent_selection": INTELLIGENT_CHUNK_SELECTION
            }
        }
        return jsonify(cache_info)
    except Exception as e:
        return jsonify({"error": f"Failed to get cache status: {str(e)}"}), 500

@app.route('/performance-metrics', methods=['GET'])
def get_performance_metrics():
    """
    Get detailed performance metrics
    """
    try:
        cache_hit_ratio = (
            PERFORMANCE_METRICS['cache_hits'] / 
            max(1, PERFORMANCE_METRICS['cache_hits'] + PERFORMANCE_METRICS['cache_misses'])
        ) * 100
        
        metrics = {
            **PERFORMANCE_METRICS,
            "cache_hit_ratio": f"{cache_hit_ratio:.1f}%",
            "cache_size": len(DATA_CACHE),
            "active_threads": threading.active_count(),
            "thread_pool_info": {
                "max_workers": thread_pool._max_workers,
                "threads_active": len([t for t in threading.enumerate() if "ChunkProcessor" in t.name])
            },
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to get performance metrics: {str(e)}"}), 500

@app.route('/')
def index():
    """
    Serve the main index.html page
    """
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "service": "Gemini Tinybird AI Agent with Streaming",
        "endpoints": {
            "ask_question": "POST /store/ask (REST) or WebSocket 'ask_question' event",
            "generate_chart": "POST /store/generate-chart",
            "get_suggestions": "GET /store/suggestions",
            "get_chat_history": "GET /store/chat-history",
            "clear_history": "POST /store/clear-history",
            "cleanup_sessions": "POST /cleanup-sessions",
            "clear_cache": "POST /clear-cache",
            "cache_status": "GET /cache-status"
        },
        "optimizations": {
            "caching_enabled": True,
            "cache_timeout": f"{CACHE_TIMEOUT}s",
            "max_data_limit": MAX_DATA_SIZE,
            "suggestion_sample_size": SUGGESTION_SAMPLE_SIZE,
            "parallel_processing": True,
            "max_parallel_chunks": MAX_PARALLEL_CHUNKS,
            "intelligent_chunk_selection": INTELLIGENT_CHUNK_SELECTION,
            "comprehensive_logging": True,
            "performance_monitoring": True
        },
        "websocket": {
            "enabled": True,
            "events": {
                "connect": "Client connection established",
                "ask_question": "Send question and receive streamed response",
                "stream_chunk": "Receive response chunks in real-time",
                "stream_complete": "Response streaming completed",
                "complete_response": "Full response data",
                "error": "Error notifications"
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', False))