"""
Query expansion for improving retrieval recall.

Generates semantic variations of user queries to match different document phrasings.
"""
from typing import List


def expand_query(query: str) -> List[str]:
    """
    Generate query variations to improve retrieval.
    
    Args:
        query: Original user query
        
    Returns:
        List of query variations including the original
    """
    variations = [query]  # Always include original
    
    query_lower = query.lower()
    
    # Semester/course queries
    if "semester" in query_lower or "sem" in query_lower:
        # Extract semester number
        for num_word, num_digit in [
            ("first", "1"), ("second", "2"), ("third", "3"),
            ("fourth", "4"), ("fifth", "5"), ("sixth", "6"),
            ("seventh", "7"), ("eighth", "8"),
        ]:
            if num_word in query_lower or f"sem {num_digit}" in query_lower or f"semester {num_digit}" in query_lower:
                # Generate variations
                variations.extend([
                    query.replace(num_word, f"semester {num_digit}"),
                    query.replace(num_word, f"sem {num_digit}"),
                    query.replace(f"semester {num_digit}", f"{num_word} semester"),
                    query.replace(f"sem {num_digit}", f"S{num_digit}"),
                ])
        
        # Subject/course synonyms
        if "subject" in query_lower:
            variations.append(query.replace("subject", "course"))
            variations.append(query.replace("subjects", "courses"))
            variations.append(query.replace("subject", "curriculum"))
        
        if "course" in query_lower:
            variations.append(query.replace("course", "subject"))
            variations.append(query.replace("courses", "subjects"))
    
    # Attendance queries
    if "attendance" in query_lower:
        variations.extend([
            query.replace("attendance", "attendance requirement"),
            query.replace("attendance", "attendance policy"),
            query.replace("attendance", "attendance rule"),
        ])
    
    # Exam queries
    if "exam" in query_lower:
        variations.extend([
            query.replace("exam", "examination"),
            query.replace("exams", "examinations"),
            query.replace("exam", "test"),
        ])
    
    # Syllabus queries
    if "syllabus" in query_lower:
        variations.extend([
            query.replace("syllabus", "curriculum"),
            query.replace("syllabus", "course outline"),
            query.replace("syllabus", "course content"),
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        v_clean = v.strip()
        if v_clean and v_clean.lower() not in seen:
            unique_variations.append(v_clean)
            seen.add(v_clean.lower())
    
    return unique_variations[:5]  # Limit to top 5 variations


def generate_semantic_variations(query: str) -> List[str]:
    """
    Generate semantic paraphrases of the query.
    
    Args:
        query: Original query
        
    Returns:
        List of semantic variations
    """
    variations = [query]
    
    query_lower = query.lower()
    
    # Common question patterns
    if query_lower.startswith("what is"):
        variations.append(query.replace("What is", "Tell me about"))
        variations.append(query.replace("What is", "Explain"))
    
    if query_lower.startswith("what are"):
        variations.append(query.replace("What are", "List"))
        variations.append(query.replace("What are", "Tell me about"))
    
    if "how many" in query_lower:
        variations.append(query.replace("How many", "What is the number of"))
        variations.append(query.replace("How many", "Number of"))
    
    if "how to" in query_lower:
        variations.append(query.replace("How to", "What is the process to"))
        variations.append(query.replace("How to", "Steps to"))
    
    # Remove duplicates
    return list(dict.fromkeys(variations))


def combine_expanded_queries(query: str) -> str:
    """
    Combine expanded queries into a single enriched query for better retrieval.
    
    Args:
        query: Original query
        
    Returns:
        Enriched query string
    """
    variations = expand_query(query)
    
    # If we have meaningful variations, combine them
    if len(variations) > 1:
        # Use the original query + key terms from variations
        key_terms = set()
        for var in variations:
            words = var.lower().split()
            key_terms.update([w for w in words if len(w) > 3])
        
        # Add key terms to original query
        enriched = f"{query} {' '.join(key_terms)}"
        return enriched
    
    return query
