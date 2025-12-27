"""
Evaluation dataset and test cases for RAG system.
"""
from typing import List, Set

from pydantic import BaseModel


class EvaluationExample(BaseModel):
    """Single evaluation test case."""
    
    query: str
    expected_answer: str
    relevant_documents: Set[str]  # Document names that should be retrieved
    category: str = "general"  # e.g., "attendance", "syllabus", "exam_rules"
    difficulty: str = "easy"  # easy, medium, hard


# Sample evaluation dataset
# In production, load from JSON/CSV
SAMPLE_EVALUATION_DATA: List[EvaluationExample] = [
    EvaluationExample(
        query="What is the minimum attendance required to sit for exams?",
        expected_answer="Students must maintain at least 75% attendance to be eligible for end-semester examinations.",
        relevant_documents={"AttendanceRules.pdf", "ExaminationPolicy.pdf"},
        category="attendance",
        difficulty="easy",
    ),
    EvaluationExample(
        query="What are the consequences of having less than 75% attendance?",
        expected_answer="Students with less than 75% attendance will not be allowed to sit for the end-semester examination and may need to repeat the course.",
        relevant_documents={"AttendanceRules.pdf"},
        category="attendance",
        difficulty="medium",
    ),
    EvaluationExample(
        query="What topics are covered in the DBMS syllabus?",
        expected_answer="The DBMS syllabus covers: Introduction to databases, ER modeling, Relational model, SQL, Normalization, Transactions, Concurrency control, and Database recovery.",
        relevant_documents={"DBMS_Syllabus.pdf", "Syllabus.pdf"},
        category="syllabus",
        difficulty="easy",
    ),
    EvaluationExample(
        query="When do the mid-term exams start and what is their weightage?",
        expected_answer="Mid-term exams are scheduled in the 8th week of the semester and carry 30% weightage of the total course grade.",
        relevant_documents={"AcademicCalendar.pdf", "ExaminationPolicy.pdf"},
        category="exam_schedule",
        difficulty="medium",
    ),
    EvaluationExample(
        query="What is the minimum CGPA required for placement eligibility and are there any other criteria?",
        expected_answer="Students must have a minimum CGPA of 6.5 with no active backlogs to be eligible for campus placements. Additionally, students must have completed all mandatory internships.",
        relevant_documents={"PlacementPolicy.pdf"},
        category="placement",
        difficulty="hard",
    ),
    EvaluationExample(
        query="What items are prohibited in the examination hall?",
        expected_answer="Prohibited items include: mobile phones, smart watches, calculators (unless specifically allowed), books, notes, and any electronic devices.",
        relevant_documents={"ExaminationRules.pdf", "ExamConduct.pdf"},
        category="exam_rules",
        difficulty="easy",
    ),
    EvaluationExample(
        query="How many times can a student appear for a re-examination?",
        expected_answer="A student can appear for re-examination a maximum of two times for each course.",
        relevant_documents={"ExaminationPolicy.pdf"},
        category="exam_rules",
        difficulty="medium",
    ),
    EvaluationExample(
        query="What is the hostel fee structure and payment deadline?",
        expected_answer="The hostel fee is ₹60,000 per semester and must be paid by the 15th of the first month of each semester.",
        relevant_documents={"HostelGuidelines.pdf", "FeeStructure.pdf"},
        category="hostel",
        difficulty="easy",
    ),
    EvaluationExample(
        query="If a student has 74% attendance and a medical certificate, can they sit for exams?",
        expected_answer="Students with medical certificates approved by the university health center may be granted attendance relaxation. The student must apply through the proper channel within 7 days of absence.",
        relevant_documents={"AttendanceRules.pdf", "MedicalLeavePolicy.pdf"},
        category="attendance",
        difficulty="hard",
    ),
    EvaluationExample(
        query="What is the process to apply for a scholarship?",
        expected_answer="Students must submit the scholarship application form along with required documents (income certificate, caste certificate if applicable, previous year mark sheets) to the scholarship cell before the deadline announced in the notice.",
        relevant_documents={"ScholarshipGuidelines.pdf"},
        category="scholarship",
        difficulty="medium",
    ),
]


def get_evaluation_dataset() -> List[EvaluationExample]:
    """Get the evaluation dataset."""
    return SAMPLE_EVALUATION_DATA


def filter_by_category(dataset: List[EvaluationExample], category: str) -> List[EvaluationExample]:
    """Filter dataset by category."""
    return [ex for ex in dataset if ex.category == category]


def filter_by_difficulty(dataset: List[EvaluationExample], difficulty: str) -> List[EvaluationExample]:
    """Filter dataset by difficulty."""
    return [ex for ex in dataset if ex.difficulty == difficulty]
