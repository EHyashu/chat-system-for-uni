"""
Batch evaluation pipeline for RAG system.

Runs evaluation on a test dataset and generates comprehensive metrics.
"""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app.config import Settings, get_settings
from app.evaluation.dataset import EvaluationExample, get_evaluation_dataset
from app.evaluation.metrics import (
    AnswerQualityMetrics,
    HallucinationDetector,
    RetrievalMetrics,
    calculate_aggregate_score,
)
from app.rag.pipeline import RAGPipeline


@dataclass
class EvaluationResult:
    """Result for a single evaluation example."""
    
    query: str
    expected_answer: str
    generated_answer: str
    relevant_documents: List[str]
    retrieved_documents: List[str]
    
    # Retrieval metrics
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    
    # Answer quality metrics
    semantic_similarity: float
    faithfulness: float
    answer_relevance: float
    
    # Hallucination detection
    is_hallucinated: bool
    hallucination_score: float
    
    # Overall
    confidence: float
    aggregate_score: float
    
    category: str
    difficulty: str


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline = RAGPipeline.from_settings(settings)
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerQualityMetrics(settings.embedding_model_name)
        self.hallucination_detector = HallucinationDetector(settings.embedding_model_name)
    
    async def evaluate_single(self, example: EvaluationExample, k: int = 5) -> EvaluationResult:
        """Evaluate a single test case."""
        
        # Get system response
        response = await self.pipeline.answer_query(example.query)
        
        generated_answer = response["answer"]
        sources = response["sources"]
        confidence = response.get("confidence", 0.0)
        
        # Extract retrieved document names
        retrieved_docs = [src["document_name"] for src in sources]
        
        # Retrieval metrics
        precision = self.retrieval_metrics.precision_at_k(
            retrieved_docs, example.relevant_documents, k
        )
        recall = self.retrieval_metrics.recall_at_k(
            retrieved_docs, example.relevant_documents, k
        )
        f1 = self.retrieval_metrics.f1_at_k(
            retrieved_docs, example.relevant_documents, k
        )
        
        # Answer quality metrics
        semantic_sim = self.answer_metrics.semantic_similarity(
            generated_answer, example.expected_answer
        )
        
        context_chunks = [src["text"] for src in sources]
        faithfulness = self.answer_metrics.faithfulness_score(
            generated_answer, context_chunks
        )
        
        relevance = self.answer_metrics.answer_relevance(
            example.query, generated_answer
        )
        
        # Hallucination detection
        hallucination_result = self.hallucination_detector.detect_hallucination(
            generated_answer, context_chunks
        )
        
        # Aggregate score
        aggregate = calculate_aggregate_score(
            retrieval_precision=precision,
            retrieval_recall=recall,
            semantic_similarity=semantic_sim,
            faithfulness=faithfulness,
            hallucination_score=hallucination_result["hallucination_score"],
        )
        
        return EvaluationResult(
            query=example.query,
            expected_answer=example.expected_answer,
            generated_answer=generated_answer,
            relevant_documents=list(example.relevant_documents),
            retrieved_documents=retrieved_docs,
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            semantic_similarity=semantic_sim,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            is_hallucinated=hallucination_result["is_hallucinated"],
            hallucination_score=hallucination_result["hallucination_score"],
            confidence=confidence,
            aggregate_score=aggregate,
            category=example.category,
            difficulty=example.difficulty,
        )
    
    async def evaluate_batch(
        self,
        examples: List[EvaluationExample],
        k: int = 5,
    ) -> Dict:
        """
        Evaluate multiple examples and return aggregate statistics.
        
        Returns:
            Dict with overall metrics and per-example results
        """
        results: List[EvaluationResult] = []
        
        for example in examples:
            try:
                result = await self.evaluate_single(example, k=k)
                results.append(result)
                print(f"✓ Evaluated: {example.query[:60]}...")
            except Exception as e:
                print(f"✗ Failed: {example.query[:60]}... | Error: {e}")
        
        if not results:
            return {"error": "No results to evaluate"}
        
        # Aggregate statistics
        avg_metrics = {
            "precision_at_k": sum(r.precision_at_k for r in results) / len(results),
            "recall_at_k": sum(r.recall_at_k for r in results) / len(results),
            "f1_at_k": sum(r.f1_at_k for r in results) / len(results),
            "semantic_similarity": sum(r.semantic_similarity for r in results) / len(results),
            "faithfulness": sum(r.faithfulness for r in results) / len(results),
            "answer_relevance": sum(r.answer_relevance for r in results) / len(results),
            "hallucination_rate": sum(r.is_hallucinated for r in results) / len(results),
            "avg_hallucination_score": sum(r.hallucination_score for r in results) / len(results),
            "avg_confidence": sum(r.confidence for r in results) / len(results),
            "aggregate_score": sum(r.aggregate_score for r in results) / len(results),
        }
        
        # Breakdown by category
        categories = {}
        for result in results:
            cat = result.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_metrics = {}
        for cat, cat_results in categories.items():
            category_metrics[cat] = {
                "count": len(cat_results),
                "avg_aggregate_score": sum(r.aggregate_score for r in cat_results) / len(cat_results),
                "avg_precision": sum(r.precision_at_k for r in cat_results) / len(cat_results),
                "avg_faithfulness": sum(r.faithfulness for r in cat_results) / len(cat_results),
            }
        
        # Breakdown by difficulty
        difficulties = {}
        for result in results:
            diff = result.difficulty
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(result)
        
        difficulty_metrics = {}
        for diff, diff_results in difficulties.items():
            difficulty_metrics[diff] = {
                "count": len(diff_results),
                "avg_aggregate_score": sum(r.aggregate_score for r in diff_results) / len(diff_results),
                "avg_semantic_similarity": sum(r.semantic_similarity for r in diff_results) / len(diff_results),
            }
        
        # Identify failures (low aggregate score)
        failures = [r for r in results if r.aggregate_score < 0.5]
        
        return {
            "total_examples": len(results),
            "overall_metrics": avg_metrics,
            "category_breakdown": category_metrics,
            "difficulty_breakdown": difficulty_metrics,
            "failures": [
                {
                    "query": f.query,
                    "expected": f.expected_answer[:100],
                    "generated": f.generated_answer[:100],
                    "aggregate_score": f.aggregate_score,
                    "reason": "Low semantic similarity" if f.semantic_similarity < 0.5 else "Possible hallucination",
                }
                for f in failures
            ],
            "detailed_results": [
                {
                    "query": r.query,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "precision": round(r.precision_at_k, 3),
                    "recall": round(r.recall_at_k, 3),
                    "semantic_sim": round(r.semantic_similarity, 3),
                    "faithfulness": round(r.faithfulness, 3),
                    "confidence": round(r.confidence, 3),
                    "aggregate": round(r.aggregate_score, 3),
                    "hallucinated": r.is_hallucinated,
                }
                for r in results
            ],
        }
    
    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save evaluation results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


async def run_evaluation(output_file: str = "evaluation_results.json") -> None:
    """Main evaluation entry point."""
    print("=" * 80)
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)
    
    settings = get_settings()
    evaluator = RAGEvaluator(settings)
    
    # Get evaluation dataset
    dataset = get_evaluation_dataset()
    print(f"\nLoaded {len(dataset)} evaluation examples")
    
    # Run batch evaluation
    print("\nRunning evaluation...\n")
    results = await evaluator.evaluate_batch(dataset, k=5)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    metrics = results["overall_metrics"]
    print(f"\n📊 Overall Metrics:")
    print(f"  Precision@5:         {metrics['precision_at_k']:.3f}")
    print(f"  Recall@5:            {metrics['recall_at_k']:.3f}")
    print(f"  F1@5:                {metrics['f1_at_k']:.3f}")
    print(f"  Semantic Similarity: {metrics['semantic_similarity']:.3f}")
    print(f"  Faithfulness:        {metrics['faithfulness']:.3f}")
    print(f"  Answer Relevance:    {metrics['answer_relevance']:.3f}")
    print(f"  Hallucination Rate:  {metrics['hallucination_rate']:.1%}")
    print(f"  Avg Confidence:      {metrics['avg_confidence']:.3f}")
    print(f"  Aggregate Score:     {metrics['aggregate_score']:.3f}")
    
    print(f"\n📂 Category Breakdown:")
    for cat, cat_metrics in results["category_breakdown"].items():
        print(f"  {cat}: {cat_metrics['avg_aggregate_score']:.3f} (n={cat_metrics['count']})")
    
    print(f"\n📈 Difficulty Breakdown:")
    for diff, diff_metrics in results["difficulty_breakdown"].items():
        print(f"  {diff}: {diff_metrics['avg_aggregate_score']:.3f} (n={diff_metrics['count']})")
    
    if results["failures"]:
        print(f"\n⚠️  Failed Examples ({len(results['failures'])}):")
        for fail in results["failures"][:3]:  # Show top 3 failures
            print(f"  - {fail['query'][:60]}... (score: {fail['aggregate_score']:.3f})")
    
    # Save results
    output_path = settings.base_dir / output_file
    evaluator.save_results(results, output_path)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
