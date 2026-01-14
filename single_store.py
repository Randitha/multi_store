import os
import json
import requests
import google.generativeai as genai
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Any, Generator
import logging

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Tinybird configuration
TINYBIRD_PIPE_URL = os.getenv('TINYBIRD_PIPE_URL')
TINYBIRD_TOKEN = os.getenv('TINYBIRD_TOKEN')

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def query_tinybird_api(params=None):
    """
    Query the Tinybird API pipe (legacy function - use query_tinybird_with_limit for better performance)
    """
    try:
        if params is None:
            params = {}

        # Add token to parameters
        params['token'] = TINYBIRD_TOKEN

        # Make request to Tinybird API
        response = requests.get(TINYBIRD_PIPE_URL, params=params)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        return {"error": f"Failed to query Tinybird API: {str(e)}"}

def query_tinybird_with_limit(limit=10000, additional_params=None):
    """
    Query the Tinybird API with a specific limit for better performance
    Default limit of 10,000 records should be sufficient for most analysis
    """
    try:
        params = {'limit': str(limit)}
        
        if additional_params:
            params.update(additional_params)

        # Add token to parameters
        params['token'] = TINYBIRD_TOKEN

        # Make request to Tinybird API
        response = requests.get(TINYBIRD_PIPE_URL, params=params)
        response.raise_for_status()

        result = response.json()
        
        # Add metadata about the query limit for transparency
        if 'statistics' not in result:
            result['statistics'] = {}
        result['statistics']['query_limit'] = limit
        result['statistics']['optimized'] = True
        
        return result
    except Exception as e:
        return {"error": f"Failed to query Tinybird API with limit: {str(e)}"}

def get_sample_data_for_suggestions(limit=50):
    """
    Get a small sample of data specifically for generating suggestions
    Uses limited records to avoid token limits and improve speed
    """
    try:
        # Use the optimized query function with specified limit
        params = {'limit': str(limit)}
        response = requests.get(TINYBIRD_PIPE_URL, params={'token': TINYBIRD_TOKEN, 'limit': str(limit)})
        response.raise_for_status()
        
        result = response.json()
        
        # Add metadata for transparency
        if 'statistics' not in result:
            result['statistics'] = {}
        result['statistics']['sample_size'] = limit
        result['statistics']['purpose'] = 'suggestions'
        
        return result
    except Exception as e:
        return {"error": f"Failed to get sample data: {str(e)}"}

def get_chart_data_sample(data, max_items=200):
    """
    Get a reasonable sample of data for chart generation
    Reduces data size while maintaining representativeness
    """
    try:
        if not isinstance(data, dict) or 'data' not in data:
            return data
            
        data_items = data['data']
        if len(data_items) <= max_items:
            return data
        
        # Take a stratified sample - get items from beginning, middle, and end
        sample_size = max_items
        step = len(data_items) // sample_size
        
        if step <= 1:
            # If step is 1 or less, just take first max_items
            sampled_items = data_items[:max_items]
        else:
            # Take every nth item to get a representative sample
            sampled_items = data_items[::step][:sample_size]
        
        return {
            'meta': data.get('meta', []),
            'data': sampled_items,
            'rows': len(sampled_items),
            'statistics': {
                **data.get('statistics', {}),
                'original_size': len(data_items),
                'sampled_size': len(sampled_items),
                'chart_optimized': True
            }
        }
    except Exception:
        # Return original data if sampling fails
        return data

def has_zero_results(question, data):
    """
    Check if a question would likely return zero results based on the data structure
    Made less aggressive to avoid false positives
    """
    try:
        # Convert data to string for analysis
        data_str = json.dumps(data).lower()

        # Only check for very specific no-data scenarios (less aggressive)
        no_data_keywords = [
            'tomorrow', 'next week', 'next month', 'next year', 'future',
            'prediction', 'forecast', 'will be', 'going to',
            '2025', '2026', '2027', '2028', '2029', '2030',  # Future years
            'specific person', 'individual', 'by name', 'personal information',
            'email', 'phone number', 'address', 'contact details'
        ]

        # Check if question contains keywords that would return no data
        question_lower = question.lower()
        for keyword in no_data_keywords:
            if keyword in question_lower:
                return True

        # Check if the data is completely empty
        if isinstance(data, dict) and 'data' in data:
            if len(data['data']) == 0:
                return True
            
            # Only return True if we're absolutely certain there's no relevant data
            # Remove the aggressive field-specific checks that were too restrictive

        return False

    except Exception:
        # If we can't determine, assume it might have results (less aggressive)
        return False

def clean_gemini_response(response_text):
    """
    Clean the Gemini response by removing table names and other technical references
    """
    # Common table names and abbreviations to remove
    table_patterns = [
        r'\bpc\.', r'\bagd\.', r'\bgd\.', r'\bpctd\.',  # Table prefixes
        r'\bpeople_counting\.', r'\bage_group_details\.', r'\bgender_details\.', r'\bpeople_counting_trend_details\.',  # Full table names
        r'\(pc\)', r'\(agd\)', r'\(gd\)', r'\(pctd\)',  # Table names in parentheses
        r'\[.*?\]',  # Any text in square brackets (often technical references)
        r'\b(table|column|field|join|query|sql)\b',  # Technical database terms
    ]

    # Remove table references
    for pattern in table_patterns:
        response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)

    # Clean up extra spaces and punctuation
    response_text = re.sub(r'\s+', ' ', response_text)  # Multiple spaces to single space
    response_text = re.sub(r'\s([.,!?;:])', r'\1', response_text)  # Space before punctuation
    response_text = re.sub(r'\.\.+', '.', response_text)  # Multiple dots to single dot
    response_text = re.sub(r',\s*,', ',', response_text)  # Multiple commas to single comma

    # Capitalize first letter and ensure proper ending
    response_text = response_text.strip()
    
    # Additional cleaning for technical field names
    response_text = re.sub(r'\\_+', ' ', response_text)  # Replace escaped underscores
    response_text = re.sub(r'_+', ' ', response_text)  # Replace underscores with spaces
    response_text = response_text.replace('\\.', '.')  # Fix escaped dots
    
    if response_text and not response_text.endswith(('.', '!', '?')):
        response_text += '.'

    return response_text

def chunk_data_for_analysis(data, max_chunk_size=50000):  # Significantly reduced for API limits
    """
    Split large data into manageable chunks for Gemini analysis
    Optimized for faster processing
    """
    if not isinstance(data, dict) or 'data' not in data:
        return [data]

    data_items = data['data']
    if not data_items:
        return [data]

    # Convert to JSON string to estimate size
    data_json = json.dumps(data_items)

    # If data is small enough, return as single chunk
    if len(data_json) <= max_chunk_size:
        return [data]

    # Calculate chunk size based on number of items
    avg_item_size = len(data_json) / len(data_items)
    items_per_chunk = max(200, int(max_chunk_size / avg_item_size))  # Increased minimum for meaningful analysis

    chunks = []
    for i in range(0, len(data_items), items_per_chunk):
        chunk_items = data_items[i:i + items_per_chunk]
        chunk = {
            'meta': data.get('meta', []),
            'data': chunk_items,
            'rows': len(chunk_items),
            'statistics': {
                **data.get('statistics', {}),
                'chunk_index': len(chunks),
                'total_chunks': (len(data_items) + items_per_chunk - 1) // items_per_chunk,
                'chunk_size': len(chunk_items)
            }
        }
        chunks.append(chunk)

    return chunks
def intelligent_chunk_selection(question, data_chunks, max_chunks=2):  # Minimal chunks to avoid API limits
    """
    Intelligently select the most relevant chunks based on the question
    Made less restrictive to ensure data availability
    """
    if len(data_chunks) <= max_chunks:
        return data_chunks
    
    try:
        question_lower = question.lower()
        
        # Score chunks based on relevance (less restrictive scoring)
        chunk_scores = []
        
        for i, chunk in enumerate(data_chunks):
            score = 5  # Start with base score instead of 0
            chunk_data = chunk.get('data', [])
            
            if not chunk_data:
                chunk_scores.append((i, score))
                continue
            
            # Convert chunk to string for analysis
            try:
                chunk_str = json.dumps(chunk_data[:10]).lower()  # Only check first 10 items for performance
            except:
                chunk_str = str(chunk_data[:10]).lower()
            
            # Time-based scoring (less restrictive)
            if any(word in question_lower for word in ['recent', 'today', 'yesterday', 'week', 'latest', 'time']):
                if any(term in chunk_str for term in ['timestamp', 'date', 'time']):
                    score += 8  # Reduced from 10
            
            # Age-based scoring
            if any(word in question_lower for word in ['age', 'young', 'old', 'adult', 'child']):
                if 'age' in chunk_str:
                    score += 10
            
            # Gender-based scoring  
            if any(word in question_lower for word in ['gender', 'male', 'female', 'man', 'woman']):
                if any(term in chunk_str for term in ['gender', 'male', 'female']):
                    score += 10
            
            # Count/total scoring (very generous)
            if any(word in question_lower for word in ['total', 'count', 'number', 'how many', 'what', 'who']):
                score += 8  # Give good score to most chunks for general questions
            
            # Trend/comparison scoring
            if any(word in question_lower for word in ['trend', 'compare', 'change', 'increase', 'decrease']):
                score += 6  # Less restrictive
            
            # Data density bonus (more generous)
            score += min(len(chunk_data) / 500, 8)  # More generous bonus
            
            chunk_scores.append((i, score))
        
        # Sort by score and select top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in chunk_scores[:max_chunks]]
        selected_chunks = [data_chunks[i] for i in selected_indices]
        
        logging.info(f"Selected {len(selected_chunks)} chunks with scores: {[score for _, score in chunk_scores[:max_chunks]]}")
        
        return selected_chunks
        
    except Exception as e:
        logging.warning(f"Chunk selection failed, using first {max_chunks} chunks: {e}")
        return data_chunks[:max_chunks]
    """
    Filter data to only include records relevant to the question
    This helps reduce processing time and improves response accuracy
    """
    try:
        if not isinstance(data, dict) or 'data' not in data:
            return data
            
        data_items = data['data']
        if len(data_items) <= max_items:
            return data
        
        question_lower = question.lower()
        
        # Time-based filtering
        if any(word in question_lower for word in ['yesterday', 'today', 'week', 'month', 'recent', 'latest']):
            # Sort by timestamp if available and take most recent
            if data_items and isinstance(data_items[0], dict):
                timestamp_fields = ['timestamp', 'time', 'date', 'created_at', 'datetime']
                timestamp_field = None
                for field in timestamp_fields:
                    if field in data_items[0]:
                        timestamp_field = field
                        break
                        
                if timestamp_field:
                    try:
                        sorted_items = sorted(data_items, 
                                            key=lambda x: x.get(timestamp_field, ''), 
                                            reverse=True)
                        filtered_items = sorted_items[:max_items]
                        
                        return {
                            'meta': data.get('meta', []),
                            'data': filtered_items,
                            'rows': len(filtered_items),
                            'statistics': {
                                **data.get('statistics', {}),
                                'filtered': True,
                                'filter_type': 'time_based',
                                'original_size': len(data_items),
                                'filtered_size': len(filtered_items)
                            }
                        }
                    except Exception:
                        pass
        
        # If no specific filtering applies, take a representative sample
        step = len(data_items) // max_items
        if step <= 1:
            filtered_items = data_items[:max_items]
        else:
            filtered_items = data_items[::step][:max_items]
        
        return {
            'meta': data.get('meta', []),
            'data': filtered_items,
            'rows': len(filtered_items),
            'statistics': {
                **data.get('statistics', {}),
                'filtered': True,
                'filter_type': 'representative_sample',
                'original_size': len(data_items),
                'filtered_size': len(filtered_items)
            }
        }
        
    except Exception:
        # Return original data if filtering fails
        return data

def process_chunk_parallel(chunk_data, question, chunk_index, total_chunks):
    """
    Process a single chunk in parallel with better size management
    Returns the result or None if processing fails
    """
    try:
        # Limit data size to avoid API errors
        sample_data = chunk_data.get('data', [])[:20]  # Very small sample
        
        chunk_prompt = f"""
        Based on this data chunk ({chunk_index + 1}/{total_chunks}) from our people counting system,
        analyze this question: {question}

        Data Sample (first 100 records of {len(chunk_data.get('data', []))} total): {json.dumps(sample_data, indent=1)}

        Provide key insights from this chunk that help answer the question.
        Focus on specific numbers, patterns, or trends.
        
        IMPORTANT:
        - If this chunk has no relevant data, respond with "NO_RELEVANT_DATA"
        - Use clear, concise language
        - Include specific numbers when available
        - Focus on business insights
        """

        # Use a shorter timeout for individual chunks
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(
            chunk_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,  # Limit output
                temperature=0.1
            )
        )
        result = response.text.strip()
        
        # Clean technical field names from the result
        result = clean_gemini_response(result)
        
        if "NO_RELEVANT_DATA" in result.upper() or "no relevant data" in result.lower():
            return None
        
        logging.info(f"Chunk {chunk_index + 1} processed successfully - {len(result)} chars")
        return result
        
    except Exception as e:
        logging.warning(f"Chunk {chunk_index + 1} processing failed: {str(e)[:100]}")
        return None

def intelligent_chunk_selection(question, data_chunks, max_chunks=3):
    """
    Intelligently select the most relevant chunks based on the question
    This dramatically reduces processing time by only analyzing relevant data
    """
    if len(data_chunks) <= max_chunks:
        return data_chunks
    
    try:
        question_lower = question.lower()
        
        # Score chunks based on relevance
        chunk_scores = []
        
        for i, chunk in enumerate(data_chunks):
            score = 0
            chunk_data = chunk.get('data', [])
            
            if not chunk_data:
                chunk_scores.append((i, score))
                continue
            
            # Convert chunk to string for analysis
            chunk_str = json.dumps(chunk_data).lower()
            
            # Time-based scoring
            if any(word in question_lower for word in ['recent', 'today', 'yesterday', 'week', 'latest']):
                # Prefer chunks with more recent timestamps
                if 'timestamp' in chunk_str or 'date' in chunk_str:
                    score += 10
            
            # Age-based scoring
            if any(word in question_lower for word in ['age', 'young', 'old', 'adult', 'child']):
                if 'age' in chunk_str:
                    score += 15
            
            # Gender-based scoring
            if any(word in question_lower for word in ['gender', 'male', 'female', 'man', 'woman']):
                if any(term in chunk_str for term in ['gender', 'male', 'female']):
                    score += 15
            
            # Count/total scoring
            if any(word in question_lower for word in ['total', 'count', 'number', 'how many']):
                # All chunks are relevant for counting
                score += 5
            
            # Trend/comparison scoring
            if any(word in question_lower for word in ['trend', 'compare', 'change', 'increase', 'decrease']):
                # Prefer chunks with varied data
                if len(set(str(item) for item in chunk_data[:10])) > 3:
                    score += 8
            
            # Data density bonus
            score += min(len(chunk_data) / 1000, 5)  # Bonus for data-rich chunks
            
            chunk_scores.append((i, score))
        
        # Sort by score and select top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in chunk_scores[:max_chunks]]
        selected_chunks = [data_chunks[i] for i in selected_indices]
        
        logging.info(f"Selected {len(selected_chunks)} most relevant chunks out of {len(data_chunks)} total chunks")
        
        return selected_chunks
        
    except Exception as e:
        logging.warning(f"Chunk selection failed, using first {max_chunks} chunks: {e}")
        return data_chunks[:max_chunks]

def analyze_data_with_parallel_processing(question, data, request_id=""):
    """
    Analyze data using parallel chunk processing for maximum speed
    Generator function that yields response chunks as they're processed
    """
    start_time = time.time()
    
    try:
        logging.info(f"[{request_id}] Starting parallel data analysis")
        
        # Log initial data size
        initial_size = len(data.get('data', [])) if isinstance(data, dict) else 0
        logging.info(f"[{request_id}] Initial data size: {initial_size} records")
        
        # Apply intelligent filtering first
        filtered_data = filter_relevant_data(question, data, max_items=1000)  # Minimal for demonstration
        
        # Log filtered data size
        filtered_size = len(filtered_data.get('data', [])) if isinstance(filtered_data, dict) else 0
        logging.info(f"[{request_id}] After filtering: {filtered_size} records")
        
        # Create chunks optimized for parallel processing
        data_chunks = chunk_data_for_analysis(filtered_data, max_chunk_size=300000)  # Larger chunks
        
        total_chunks = len(data_chunks)
        logging.info(f"[{request_id}] Created {total_chunks} chunks for parallel processing")
        
        if total_chunks == 1:
            # Single chunk - process directly with streaming
            logging.info(f"[{request_id}] Processing single chunk directly")
            
            prompt = f"""
            Based on the following data from our people counting system, answer this question: {question}

            Data: {json.dumps(data_chunks[0], indent=2)}

            Provide a comprehensive, well-structured answer.
            
            FORMATTING:
            - Use **bold** for key points
            - Use bullet points where appropriate
            - Include specific numbers and insights
            - Structure your response clearly
            
            IMPORTANT: Use plain English, focus on business insights.
            """

            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        
        # Multiple chunks - use intelligent selection and parallel processing
        selected_chunks = intelligent_chunk_selection(question, data_chunks, max_chunks=4)
        
        logging.info(f"[{request_id}] Selected {len(selected_chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        chunk_results = []
        with ThreadPoolExecutor(max_workers=min(4, len(selected_chunks)), thread_name_prefix=f"Chunk-{request_id}") as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_chunk_parallel, chunk, question, i, len(selected_chunks)): i
                for i, chunk in enumerate(selected_chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk, timeout=30):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result(timeout=10)
                    if result:
                        chunk_results.append(result)
                        logging.info(f"[{request_id}] Chunk {chunk_index + 1} result collected")
                except (TimeoutError, Exception) as e:
                    logging.warning(f"[{request_id}] Chunk {chunk_index + 1} failed: {e}")
                    continue
        
        processing_time = time.time() - start_time
        logging.info(f"[{request_id}] Parallel processing completed in {processing_time:.2f}s")
        
        # If no results, try with less restrictive filtering
        if not chunk_results:
            logging.warning(f"[{request_id}] No chunk results, trying with full dataset")
            # Try with much larger data sample as fallback
            fallback_data = filter_relevant_data(question, data, max_items=500)  # Very small for demonstration
            fallback_size = len(fallback_data.get('data', [])) if isinstance(fallback_data, dict) else 0
            logging.info(f"[{request_id}] Fallback data size: {fallback_size} records")
            
            if fallback_size > 0:
                # Process a single chunk with more data
                try:
                    chunk_prompt = f"""
                    Based on this data from our people counting system, please answer: {question}
                    
                    Data: {json.dumps(fallback_data, indent=2)}
                    
                    Provide a comprehensive analysis of the available data to answer this question.
                    Focus on extracting insights from whatever data is available.
                    """
                    
                    response = model.generate_content(chunk_prompt)
                    if response.text and "no relevant data" not in response.text.lower():
                        # Clean the response of technical field names
                        cleaned_response = clean_gemini_response(response.text)
                        for char in cleaned_response:
                            yield char
                        return
                except Exception as e:
                    logging.error(f"[{request_id}] Fallback processing failed: {e}")
            
            yield "I don't have enough relevant data to answer this question based on the current dataset."
            return
        
        # If single result, return it directly
        if len(chunk_results) == 1:
            for char in chunk_results[0]:
                yield char
            return
        
        # Combine multiple results using streaming
        synthesis_prompt = f"""
        Question: {question}

        I analyzed multiple data chunks in parallel and found these insights:

        {chr(10).join([f"Analysis {i+1}: {result}" for i, result in enumerate(chunk_results)])}

        Synthesize these findings into a comprehensive, well-structured answer.
        
        REQUIREMENTS:
        - Create a coherent narrative from the parallel insights
        - Use **bold** for key findings
        - Include specific numbers where available
        - Structure with clear sections if appropriate
        - Focus on the most important patterns and insights
        
        IMPORTANT: Provide a business-focused answer in plain English.
        """

        try:
            synthesis_response = model.generate_content(synthesis_prompt, stream=True)
            yield f"\n\n**Analysis Summary** (processed {len(selected_chunks)} data chunks in parallel):\n\n"
            for chunk in synthesis_response:
                if chunk.text:
                    # Clean technical field names from the streaming response
                    cleaned_chunk = clean_gemini_response(chunk.text)
                    yield cleaned_chunk
        except Exception as e:
            logging.error(f"[{request_id}] Synthesis failed: {e}")
            # Fallback: return the best single result with cleaning
            cleaned_fallback = clean_gemini_response(chunk_results[0])
            yield f"\n\n**Key Findings** (from parallel analysis):\n\n{cleaned_fallback}"
    
    except Exception as e:
        logging.error(f"[{request_id}] Parallel processing failed completely: {e}")
        yield f"Error in parallel processing: {str(e)}"

def filter_relevant_data(question, data, max_items=15000):  # Increased limit
    """
    Filter data to only include records relevant to the question
    Made less aggressive to ensure data availability
    """
    try:
        if not isinstance(data, dict) or 'data' not in data:
            return data
            
        data_items = data['data']
        if len(data_items) <= max_items:
            return data  # Return all data if under limit
        
        question_lower = question.lower()
        
        # Time-based filtering - only if very specific time requests
        if any(word in question_lower for word in ['yesterday', 'today', 'last hour']):
            # Only filter for very specific recent time requests
            if data_items and isinstance(data_items[0], dict):
                timestamp_fields = ['timestamp', 'time', 'date', 'created_at', 'datetime']
                timestamp_field = None
                for field in timestamp_fields:
                    if field in data_items[0]:
                        timestamp_field = field
                        break
                        
                if timestamp_field:
                    try:
                        sorted_items = sorted(data_items, 
                                            key=lambda x: x.get(timestamp_field, ''), 
                                            reverse=True)
                        filtered_items = sorted_items[:max_items]
                        
                        return {
                            'meta': data.get('meta', []),
                            'data': filtered_items,
                            'rows': len(filtered_items),
                            'statistics': {
                                **data.get('statistics', {}),
                                'filtered': True,
                                'filter_type': 'time_based',
                                'original_size': len(data_items),
                                'filtered_size': len(filtered_items)
                            }
                        }
                    except Exception:
                        pass
        
        # For other questions, take a larger representative sample
        # Use a more generous sampling approach
        step = max(1, len(data_items) // max_items)
        if step <= 2:
            filtered_items = data_items[:max_items]  # Take first items if step is small
        else:
            # Take every nth item but ensure we get a good distribution
            filtered_items = []
            for i in range(0, len(data_items), step):
                filtered_items.append(data_items[i])
                if len(filtered_items) >= max_items:
                    break
        
        return {
            'meta': data.get('meta', []),
            'data': filtered_items,
            'rows': len(filtered_items),
            'statistics': {
                **data.get('statistics', {}),
                'filtered': True,
                'filter_type': 'representative_sample',
                'original_size': len(data_items),
                'filtered_size': len(filtered_items)
            }
        }
        
    except Exception:
        # Return original data if filtering fails
        return data

def analyze_data_with_gemini_stream(question, data):
    """
    Use Gemini to analyze the data and stream the response in real-time
    Generator function that yields chunks of text as they're generated
    Optimized with intelligent data filtering for better performance
    """
    try:
        logging.info("Starting streaming data analysis (fallback mode)")
        
        # Apply intelligent filtering to reduce data size (less aggressive for streaming)
        filtered_data = filter_relevant_data(question, data, max_items=12000)  # Increased for better data availability
        
        # Use the filtered data for chunking
        data_chunks = chunk_data_for_analysis(filtered_data, max_chunk_size=200000)  # Reasonable chunk size

        if len(data_chunks) == 1:
            # Single chunk - process normally with streaming
            dataset_info = ""
            if 'statistics' in filtered_data:
                stats = filtered_data['statistics']
                if stats.get('filtered'):
                    dataset_info = f"\n\nNote: This analysis is based on {stats.get('filtered_size', 'a subset')} records from your dataset (optimized for performance)."
            
            prompt = f"""
            Based on the following data from our people counting system, please answer this question: {question}

            Data: {json.dumps(data_chunks[0], indent=2)}

            Please provide a clear, concise answer focusing on the key insights.
            If the data doesn't contain information to answer the question, politely state that.
            {dataset_info}

            FORMATTING RULES:
            1. Use markdown formatting for emphasis:
               - Use **text** for bold/important points
               - Start new sections with **Section Title:** on its own line
               - Use bullet points with * for lists
            2. Use clear paragraph breaks for readability
            3. Structure your response with clear sections when appropriate
            
            IMPORTANT: In your response, use only plain English without any technical references,
            table names, or column names. Focus on the business insights, not the technical implementation.
            """

            # Use Gemini's streaming API
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        else:
            # Multiple chunks - process each and combine results with streaming
            chunk_answers = []

            for i, chunk in enumerate(data_chunks):
                chunk_prompt = f"""
                Based on this partial dataset from our people counting system (chunk {i+1}/{len(data_chunks)}),
                please analyze this specific question: {question}

                Partial Data: {json.dumps(chunk, indent=2)}

                Provide key insights or observations from this data chunk that are relevant to the question.
                Focus on patterns, trends, or statistics that help answer the question.
                
                FORMATTING RULES:
                1. Use markdown formatting for emphasis:
                   - Use **text** for bold/important points
                   - Use bullet points with * for lists
                2. Use clear paragraph breaks
                
                IMPORTANT: If this chunk doesn't contain relevant information, simply state "No relevant data in this chunk".
                Use only plain English without technical references.
                """

                try:
                    chunk_response = model.generate_content(chunk_prompt)
                    chunk_answer = chunk_response.text

                    # Only include chunks with meaningful data
                    if "no relevant data" not in chunk_answer.lower():
                        chunk_answers.append(chunk_answer)

                except Exception:
                    # Continue with other chunks if one fails
                    continue

            # If no chunks produced meaningful answers
            if not chunk_answers:
                yield "I don't have enough relevant data to answer this question based on the current dataset."
                return

            # Combine insights from all chunks
            if len(chunk_answers) == 1:
                for char in chunk_answers[0]:
                    yield char
                return

            # Add dataset info if data was filtered
            dataset_info = ""
            if 'statistics' in filtered_data and filtered_data['statistics'].get('filtered'):
                stats = filtered_data['statistics']
                dataset_info = f"\n\nNote: This analysis is based on {stats.get('filtered_size', 'a subset')} records optimized for your question."

            # Use Gemini to synthesize the chunk answers with streaming
            synthesis_prompt = f"""
            Question: {question}

            I have analyzed this question across multiple data chunks and found these insights:

            {chr(10).join([f"Chunk {i+1}: {answer}" for i, answer in enumerate(chunk_answers)])}

            Please synthesize these insights into a single, coherent answer to the original question.
            Focus on the most important patterns and provide a clear, concise response.
            {dataset_info}
            
            FORMATTING RULES:
            1. Use markdown formatting for emphasis:
               - Use **text** for bold/important points
               - Start sections with **Section Title:** on new lines
               - Use bullet points with * for lists
            2. Use clear paragraph breaks for readability

            IMPORTANT: Use only plain English without technical references. Provide a business-focused answer.
            """

            try:
                synthesis_response = model.generate_content(synthesis_prompt, stream=True)
                for chunk in synthesis_response:
                    if chunk.text:
                        yield chunk.text
            except Exception:
                # Fallback: return the first meaningful chunk answer
                for char in chunk_answers[0]:
                    yield char

    except Exception as e:
        yield f"Error analyzing data with Gemini: {str(e)}"
def generate_chart_with_gemini(question, data, answer):
    """
    Use Gemini to generate an HTML chart based on the question, data, and answer
    Enhanced to support all chart types: bar, column, line, pie, doughnut, area
    """
    try:
        # First check if we have enough data to generate a meaningful chart
        if has_zero_results(question, data):
            return "<div class='chart-error'>No data available to generate a chart for this question.</div>"

        # Use a smaller data sample for chart generation to avoid token limits
        chart_data_sample = get_chart_data_sample(data, max_items=300)  # Increased slightly for better charts

        prompt = f"""
        Based on the following question, data sample, and answer, generate an HTML chart using Chart.js.

        Question: {question}
        Data Sample: {json.dumps(chart_data_sample, indent=2)}
        Answer: {answer}

        Create an appropriate chart that visualizes the key insights from the answer.
        Choose the BEST chart type based on the data and question:
        
        - BAR CHART: For comparing categories (horizontal bars)
        - COLUMN CHART: For comparing values across categories (vertical bars, use 'bar' type with indexAxis: 'x')
        - LINE CHART: For trends over time or continuous data
        - PIE CHART: For showing parts of a whole (percentage/proportion data)
        - DOUGHNUT CHART: Similar to pie but with center hole (use 'doughnut' type)
        - AREA CHART: For showing volume/magnitude over time (line chart with filled area)
        
        The chart should be responsive and well-styled.

        IMPORTANT REQUIREMENTS:
        1. Return ONLY the HTML code with embedded JavaScript for the chart.
        2. Use Chart.js version 3.x or 4.x syntax - DO NOT include the Chart.js CDN script
        3. Use a UNIQUE canvas ID: "chart_" + random 6-digit number
        4. Make the chart RESPONSIVE with proper configuration
        5. Use appropriate colors and labels that match the question
        6. Add a meaningful title that summarizes the insight
        7. Choose the chart type that BEST represents the data
        8. Include proper chart options for responsiveness
        9. Do NOT include any explanations outside the HTML code
        10. Ensure canvas has proper wrapper div with responsive styles

        CRITICAL CHART.JS SYNTAX:
        - For responsive charts, use: responsive: true, maintainAspectRatio: true
        - For bar charts (vertical): type: 'bar', no indexAxis needed (default is 'x')
        - For horizontal bars: type: 'bar', options: {{ indexAxis: 'y' }}
        - For doughnut: type: 'doughnut'
        - For area charts: type: 'line' with fill: true in dataset
        - Always wrap canvas in a container div with proper styling

        Example structure (adapt based on chosen chart type):
        <div style="width: 100%; max-width: 700px; margin: 20px auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <canvas id="chart_XXXXXX" style="width: 100%; height: 400px;"></canvas>
        </div>
        <script>
        (function() {{
            const ctx = document.getElementById('chart_XXXXXX');
            if (!ctx) return;
            
            // Destroy existing chart if any
            const existingChart = Chart.getChart(ctx);
            if (existingChart) existingChart.destroy();
            
            new Chart(ctx, {{
                type: 'bar', // or 'line', 'pie', 'doughnut'
                data: {{
                    labels: ['Label1', 'Label2'],
                    datasets: [{{
                        label: 'Dataset Label',
                        data: [value1, value2],
                        backgroundColor: ['#4f46e5', '#8b5cf6'],
                        borderColor: ['#4338ca', '#7c3aed'],
                        borderWidth: 2,
                        fill: false // set to true for area charts
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Chart Title',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            enabled: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }})();
        </script>
        
        Generate a complete, working chart that displays properly in the browser.
        """

        response = model.generate_content(prompt)

        # Extract the HTML code from the response
        html_code = response.text.strip()

        # Clean up the response to ensure it's valid HTML
        if html_code.startswith('```html'):
            html_code = html_code[7:]
        if html_code.endswith('```'):
            html_code = html_code[:-3]
        if html_code.startswith('```'):
            html_code = html_code[3:]
        if html_code.endswith('```'):
            html_code = html_code[:-3]

        html_code = html_code.strip()

        # Remove any Chart.js CDN script tags
        html_code = re.sub(r'<script[^>]*src=["\'][^"\']*chart[^"\']*["\'][^>]*>.*?</script>', '', html_code, flags=re.IGNORECASE | re.DOTALL)

        # Ensure we have a proper canvas element
        if '<canvas' not in html_code:
            html_code = '''
            <div style="width: 100%; max-width: 700px; margin: 20px auto; padding: 20px; background: white; border-radius: 12px;">
                <p style="text-align: center; color: #666;">Unable to generate chart visualization</p>
            </div>
            '''

        return html_code

    except Exception as e:
        return f"<div class='chart-error'>Error generating chart: {str(e)}</div>"