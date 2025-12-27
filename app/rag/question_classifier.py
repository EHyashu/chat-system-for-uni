"""
Question classifier to distinguish between university-specific and general queries.
"""
import re
from typing import Literal


QuestionType = Literal["university", "general_greeting", "general_question", "out_of_scope"]


def classify_question(query: str) -> QuestionType:
    """
    Classify the question type to determine how to handle it.
    
    Args:
        query: User's question
        
    Returns:
        Question type: university, general_greeting, general_question, out_of_scope
    """
    query_lower = query.lower().strip()
    
    # 1. Greetings and chitchat
    greetings = [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "what's up", "sup",
        "greetings", "hola", "namaste"
    ]
    if any(query_lower.startswith(g) for g in greetings):
        return "general_greeting"
    
    if query_lower in ["thanks", "thank you", "bye", "goodbye", "ok", "okay"]:
        return "general_greeting"
    
    # 2. University-specific keywords
    university_keywords = [
        "semester", "sem", "course", "subject", "syllabus", "curriculum",
        "attendance", "exam", "examination", "test", "marks", "grade",
        "cgpa", "gpa", "credits", "faculty", "professor", "teacher",
        "hostel", "admission", "eligibility", "placement", "internship",
        "fee", "scholarship", "library", "lab", "practical",
        "assignment", "project", "viva", "sessional", "mid-term",
        "end-semester", "backlog", "reappear", "university", "college",
        "campus", "department", "dean", "hod", "principal",
        "timetable", "schedule", "academic", "calendar", "notice",
        "circular", "regulation", "rule", "policy", "guideline"
    ]
    
    # Check if query contains university keywords
    if any(keyword in query_lower for keyword in university_keywords):
        return "university"
    
    # 3. Generic academic questions (could be university-specific)
    academic_patterns = [
        r"what (is|are) the .*(for|in|of)",
        r"when (is|are|do|does)",
        r"how (many|much|to|can|do)",
        r"list .* (course|subject|topic)",
        r"explain .* (policy|rule|procedure)",
        r"(minimum|maximum) .* (required|needed)",
    ]
    
    if any(re.search(pattern, query_lower) for pattern in academic_patterns):
        return "university"
    
    # 4. General knowledge questions
    general_patterns = [
        r"what is .*(python|java|programming|algorithm|data structure)",
        r"explain .*(concept|theory|principle)",
        r"how does .* work",
        r"difference between .* and",
        r"who (is|was|invented|created)",
        r"when (was|did|were) .* (invented|created|discovered)",
    ]
    
    if any(re.search(pattern, query_lower) for pattern in general_patterns):
        return "general_question"
    
    # 5. Very short queries (likely general)
    if len(query_lower.split()) <= 2 and query_lower not in ["hi", "hello", "hey"]:
        return "general_question"
    
    # Default: treat as university question
    return "university"


def should_use_retrieval(question_type: QuestionType) -> bool:
    """
    Determine if we should use RAG retrieval for this question type.
    
    Args:
        question_type: Classified question type
        
    Returns:
        True if should use retrieval, False otherwise
    """
    return question_type == "university"


def get_fallback_response(question_type: QuestionType, query: str) -> dict:
    """
    Generate appropriate fallback response based on question type.
    
    Args:
        question_type: Classified question type
        query: Original user query
        
    Returns:
        Response dict with answer and metadata
    """
    if question_type == "general_greeting":
        return {
            "answer": (
                "Hello! 👋 I'm your University AI Assistant.\n\n"
                "I can help you with questions about:\n"
                "- Course syllabus and curriculum\n"
                "- Attendance and examination policies\n"
                "- Placement eligibility and procedures\n"
                "- Hostel and admission guidelines\n"
                "- Academic calendar and schedules\n"
                "- University rules and regulations\n\n"
                "What would you like to know about your university?"
            ),
            "sources": [],
            "confidence": 1.0,
            "reasoning": "Greeting detected - providing welcome message.",
        }
    
    elif question_type == "general_question":
        return {
            "answer": (
                "I'm specifically designed to answer questions about your university's policies, "
                "courses, and regulations using official university documents.\n\n"
                "For general knowledge questions, I recommend:\n"
                "- Using a general-purpose AI like ChatGPT or Google\n"
                "- Checking educational resources specific to your topic\n\n"
                "However, if your question is related to university academics, "
                "please rephrase it to include specific university context, and I'll be happy to help!"
            ),
            "sources": [],
            "confidence": 0.0,
            "reasoning": "General question detected - outside university document scope.",
        }
    
    else:  # out_of_scope
        return {
            "answer": "I could not find this information in the university documents.",
            "sources": [],
            "confidence": 0.0,
            "reasoning": "No relevant context found in university documents.",
        }
