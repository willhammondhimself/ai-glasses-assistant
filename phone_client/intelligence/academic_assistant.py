"""
Academic Assistant - Advanced intelligence for homework and studying.

Capabilities:
1. Concept Bridge Builder - Connect ideas across courses
2. Derivation Explainer - Step-by-step math proofs
3. Problem Strategy Generator - Attack patterns for problem types
4. Exam Pattern Analyzer - Study what's likely to be tested
5. Notation Decoder - Explain unfamiliar math symbols
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import re


@dataclass
class ConceptBridge:
    """Connection between concepts across subjects."""
    concept: str
    connects_to: List[str]
    relationships: List[Dict[str, str]]  # [{'target': 'linear algebra', 'relationship': '...', 'example': '...'}]
    importance: str
    sources: List[str]
    cost: float


@dataclass
class DerivationSteps:
    """Step-by-step mathematical derivation."""
    equation: str
    context: str
    steps: List[Dict[str, str]]  # [{'expression': '...', 'explanation': '...', 'justification': '...'}]
    final_result: str
    sources: List[str]
    cost: float


@dataclass
class ProblemStrategy:
    """Attack pattern for solving a problem type."""
    problem_type: str
    approach_steps: List[str]
    key_insights: List[str]
    common_pitfalls: List[str]
    example_problem: str
    sources: List[str]
    cost: float


@dataclass
class ExamPattern:
    """Analysis of exam question patterns."""
    course: str
    exam_type: str  # 'midterm', 'final', 'quiz'
    high_frequency_topics: List[str]
    common_question_types: List[str]
    study_recommendations: List[str]
    confidence: float
    sources: List[str]
    cost: float


@dataclass
class NotationExplanation:
    """Explanation of mathematical notation."""
    notation: str
    name: str
    meaning: str
    common_contexts: List[str]
    example_usage: str
    sources: List[str]
    cost: float


@dataclass
class VisualIntuition:
    """Visual learning resources for a concept."""
    concept: str
    analogies: List[str]
    mental_models: List[str]
    visual_resources: List[Dict[str, str]]  # [{'type': 'video', 'description': '...', 'source': '...'}]
    learning_approach: str
    sources: List[str]
    cost: float


@dataclass
class FormulaSheet:
    """Organized formula reference sheet."""
    topic: str
    categories: List[Dict[str, Any]]  # [{'name': 'Derivatives', 'formulas': ['f\'(x) = ...', ...]}]
    key_relationships: List[str]
    common_mistakes: List[str]
    sources: List[str]
    cost: float


@dataclass
class PaperSummary:
    """Executive summary of academic paper."""
    title: str
    authors: str
    year: str
    main_contribution: str
    key_findings: List[str]
    methodology: str
    limitations: List[str]
    relevance_score: float  # 0.0-1.0
    sources: List[str]
    cost: float


class AcademicAssistant:
    """
    Advanced academic intelligence for homework and studying.

    Capabilities:
    1. Concept Bridge Builder - Connect ideas across courses
    2. Derivation Explainer - Step-by-step math proofs
    3. Problem Strategy Generator - Attack patterns for problem types
    4. Exam Pattern Analyzer - Study what's likely to be tested
    5. Notation Decoder - Explain unfamiliar math symbols
    """

    def __init__(self):
        from phone_client.api_clients.perplexity_client import PerplexityClient
        self.perplexity = PerplexityClient()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 7200  # 2 hours for academic content

    async def build_concept_bridge(
        self,
        concept: str,
        connect_to: List[str],
        level: str = "undergraduate"
    ) -> ConceptBridge:
        """
        Explain how a concept connects to other subjects/topics.

        Example:
            bridge = await assistant.build_concept_bridge(
                concept="Hermitian operators",
                connect_to=["linear algebra", "quantum mechanics", "functional analysis"],
                level="undergraduate"
            )

        Returns connections with specific examples and importance ratings.
        """
        cache_key = f"bridge:{concept}:{','.join(sorted(connect_to))}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        query = f"""Explain "{concept}" and how it connects to the following topics:
{chr(10).join(f'- {c}' for c in connect_to)}

For each connection, provide:
1. RELATIONSHIP: How the concepts are related (2-3 sentences)
2. EXAMPLE: Specific concrete example showing the connection
3. IMPORTANCE: Why understanding this connection matters

Keep explanations at {level} level.
Format each connection clearly with headers.
"""

        response = await self.perplexity.query(
            query,
            model='sonar-pro',
            search_domain_filter=['.edu', 'arxiv.org', 'stackexchange.com']
        )

        # Parse response into structured format
        relationships = self._parse_concept_connections(response.content, connect_to)

        result = ConceptBridge(
            concept=concept,
            connects_to=connect_to,
            relationships=relationships,
            importance=self._extract_importance(response.content),
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def explain_derivation(
        self,
        equation: str,
        context: str = "",
        show_all_steps: bool = True
    ) -> DerivationSteps:
        """
        Get step-by-step mathematical derivation.

        Example:
            derivation = await assistant.explain_derivation(
                equation="∇²ψ + k²ψ = 0",
                context="Helmholtz equation from wave equation"
            )
        """
        cache_key = f"deriv:{equation}:{context}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        detail_level = "all intermediate steps" if show_all_steps else "key steps only"

        query = f"""Provide detailed step-by-step derivation for: {equation}

Context: {context}

Show the derivation with {detail_level}. For each step:
1. Show the mathematical expression
2. Explain what's happening in this step
3. Justify why this transformation is valid

Number each step clearly.
End with the final result and briefly explain its significance.
"""

        response = await self.perplexity.query(
            query,
            model='sonar-pro',
            search_domain_filter=['.edu', 'math.stackexchange.com', 'physics.stackexchange.com']
        )

        # Parse into structured steps
        steps = self._parse_derivation_steps(response.content)

        # Extract final result from content or use last step
        final_result = self._extract_section(response.content, "final.?result|conclusion|therefore") or \
                      (steps[-1]['expression'] if steps else equation)

        result = DerivationSteps(
            equation=equation,
            context=context,
            steps=steps,
            final_result=final_result,
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def generate_problem_strategy(
        self,
        problem_type: str,
        include_examples: bool = True
    ) -> ProblemStrategy:
        """
        Generate attack pattern for solving a type of problem.

        Example:
            strategy = await assistant.generate_problem_strategy(
                problem_type="integration by parts",
                include_examples=True
            )
        """
        cache_key = f"strategy:{problem_type}:{include_examples}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        examples_instruction = "Include a concrete example problem." if include_examples else ""

        query = f"""Explain the general approach for solving: {problem_type}

Provide:
1. STEP-BY-STEP APPROACH: Numbered steps for solving this problem type
2. KEY INSIGHTS: Important concepts or tricks students should know
3. COMMON PITFALLS: Typical mistakes and how to avoid them

{examples_instruction}
Focus on the general problem-solving strategy.
"""

        response = await self.perplexity.query(
            query,
            model='sonar-pro',
            search_domain_filter=['.edu', 'khanacademy.org', 'brilliant.org']
        )

        # Parse response
        strategy_data = self._parse_problem_strategy(response.content)

        result = ProblemStrategy(
            problem_type=problem_type,
            approach_steps=strategy_data['approach_steps'],
            key_insights=strategy_data['key_insights'],
            common_pitfalls=strategy_data['common_pitfalls'],
            example_problem=strategy_data['example_problem'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def analyze_exam_patterns(
        self,
        course: str,
        exam_type: str = "final",
        past_exams_context: str = ""
    ) -> ExamPattern:
        """
        Analyze typical exam patterns for a course.

        Example:
            pattern = await assistant.analyze_exam_patterns(
                course="Linear Algebra",
                exam_type="final",
                past_exams_context="Covers eigenvalues, SVD, least squares"
            )
        """
        cache_key = f"exam:{course}:{exam_type}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        context_note = f"\nContext from past exams: {past_exams_context}" if past_exams_context else ""

        query = f"""Analyze typical {exam_type} exam patterns for: {course}{context_note}

Provide:
1. HIGH-FREQUENCY TOPICS: Most commonly tested topics
2. COMMON QUESTION TYPES: Typical question formats and styles
3. STUDY RECOMMENDATIONS: How to prepare effectively and what to prioritize

Base analysis on typical university-level exams for this course.
Include confidence estimate (0.0-1.0) for accuracy of pattern analysis.
"""

        response = await self.perplexity.query(
            query,
            model='sonar-pro',
            search_domain_filter=['.edu']
        )

        # Parse response
        pattern_data = self._parse_exam_pattern(response.content)

        result = ExamPattern(
            course=course,
            exam_type=exam_type,
            high_frequency_topics=pattern_data['high_frequency_topics'],
            common_question_types=pattern_data['common_question_types'],
            study_recommendations=pattern_data['study_recommendations'],
            confidence=pattern_data['confidence'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def decode_notation(
        self,
        notation: str,
        subject_context: str = ""
    ) -> NotationExplanation:
        """
        Explain unfamiliar mathematical notation.

        Example:
            explanation = await assistant.decode_notation(
                notation="⊗",
                subject_context="tensor products in quantum mechanics"
            )
        """
        cache_key = f"notation:{notation}:{subject_context}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        context_note = f" in the context of {subject_context}" if subject_context else ""

        query = f"""Explain the mathematical notation: {notation}{context_note}

Provide:
1. NAME: Official name(s) for this symbol
2. MEANING: What it represents mathematically
3. COMMON CONTEXTS: Where this notation commonly appears (e.g., linear algebra, calculus)
4. EXAMPLE USAGE: Show 1-2 concrete examples of this notation being used

Be precise but accessible to students.
"""

        response = await self.perplexity.query(
            query,
            model='sonar',  # Faster model for notation lookups
            search_domain_filter=['.edu', 'mathworld.wolfram.com', 'stackexchange.com']
        )

        # Parse response
        notation_data = self._parse_notation(response.content)

        result = NotationExplanation(
            notation=notation,
            name=notation_data['name'],
            meaning=notation_data['meaning'],
            common_contexts=notation_data['common_contexts'],
            example_usage=notation_data['example_usage'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def find_visual_intuition(
        self,
        concept: str,
        subject_area: str = ""
    ) -> VisualIntuition:
        """
        Find analogies, mental models, and visual resources for understanding a concept.

        Example:
            intuition = await assistant.find_visual_intuition(
                concept="Fourier Transform",
                subject_area="Signal Processing"
            )

        Returns visual learning resources with analogies and mental models.
        """
        cache_key = f"visual:{concept}:{subject_area}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        subject_note = f" in the context of {subject_area}" if subject_area else ""

        query = f"""Explain "{concept}"{subject_note} using visual intuition and analogies.

Provide:
1. ANALOGIES: 2-3 real-world analogies that make this concept concrete
2. MENTAL MODELS: Concrete mental models to understand this concept
3. VISUAL RESOURCES: Types of diagrams/visualizations that help (e.g., animations, graphs, 3D models)
4. LEARNING APPROACH: Best strategy for visual understanding

Make abstract concepts concrete and visual. Focus on intuitive explanations."""

        response = await self.perplexity.query(
            query,
            model='sonar',
            search_domain_filter=['.edu', 'brilliant.org', '3blue1brown.com', 'khanacademy.org']
        )

        visual_data = self._parse_visual_intuition(response.content)

        result = VisualIntuition(
            concept=concept,
            analogies=visual_data['analogies'],
            mental_models=visual_data['mental_models'],
            visual_resources=visual_data['visual_resources'],
            learning_approach=visual_data['learning_approach'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def generate_formula_sheet(
        self,
        topic: str
    ) -> FormulaSheet:
        """
        Generate organized formula sheet with LaTeX formatting.

        Example:
            sheet = await assistant.generate_formula_sheet(
                topic="Calculus - Integration"
            )

        Returns organized formula reference with categories and relationships.
        """
        cache_key = f"formulas:{topic}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        query = f"""Create comprehensive formula sheet for: {topic}

Provide:
1. CATEGORIES: Group formulas by category (e.g., Derivatives, Integrals, Trigonometry)
2. FORMULAS: Key formulas in LaTeX notation with brief description
3. KEY RELATIONSHIPS: How formulas connect to each other
4. COMMON MISTAKES: Typical errors students make with these formulas

Format all mathematical expressions in LaTeX notation.
Be comprehensive but focus on the most important formulas."""

        response = await self.perplexity.query(
            query,
            model='sonar',
            search_domain_filter=['.edu', 'mathworld.wolfram.com', 'khanacademy.org']
        )

        formula_data = self._parse_formula_sheet(response.content)

        result = FormulaSheet(
            topic=topic,
            categories=formula_data['categories'],
            key_relationships=formula_data['key_relationships'],
            common_mistakes=formula_data['common_mistakes'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    async def preread_paper(
        self,
        paper_title: str,
        authors: str = "",
        year: str = ""
    ) -> PaperSummary:
        """
        Get executive summary of academic paper.

        Example:
            summary = await assistant.preread_paper(
                paper_title="Attention Is All You Need",
                authors="Vaswani et al.",
                year="2017"
            )

        Returns structured summary with key findings and methodology.
        """
        cache_key = f"paper:{paper_title}:{authors}:{year}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached

        author_note = f" by {authors}" if authors else ""
        year_note = f" ({year})" if year else ""

        query = f"""Executive summary of academic paper: "{paper_title}"{author_note}{year_note}

Provide:
1. MAIN CONTRIBUTION: Core contribution in 2-3 sentences
2. KEY FINDINGS: Major results (bulleted list)
3. METHODOLOGY: Brief description of the approach used
4. LIMITATIONS: Known limitations or areas for future work
5. RELEVANCE SCORE: Significance/impact on a scale of 0.0 to 1.0

Focus on actionable insights for researchers and students.
Be concise but comprehensive."""

        response = await self.perplexity.query(
            query,
            model='sonar',
            search_domain_filter=['.edu', 'arxiv.org', 'scholar.google.com']
        )

        paper_data = self._parse_paper_summary(response.content)

        result = PaperSummary(
            title=paper_title,
            authors=authors or paper_data.get('authors', 'Unknown'),
            year=year or paper_data.get('year', ''),
            main_contribution=paper_data['main_contribution'],
            key_findings=paper_data['key_findings'],
            methodology=paper_data['methodology'],
            limitations=paper_data['limitations'],
            relevance_score=paper_data['relevance_score'],
            sources=response.citations,
            cost=response.cost
        )

        self._cache[cache_key] = (result, time.time())
        return result

    # Helper parsing methods
    def _parse_concept_connections(self, content: str, connect_to: List[str]) -> List[Dict[str, str]]:
        """Extract relationship data from response."""
        relationships = []

        try:
            # Simple parsing - look for connection patterns
            for topic in connect_to:
                # Extract sections mentioning this topic
                relationship = self._extract_section(content, f"relationship.*{topic}", multiline=True)
                example = self._extract_section(content, f"example.*{topic}", multiline=True)

                if relationship or example:
                    relationships.append({
                        'target': topic,
                        'relationship': relationship if relationship else "Connection explained in content",
                        'example': example if example else "See sources for examples"
                    })

            # If no structured matches, create generic connections
            if not relationships:
                for topic in connect_to:
                    relationships.append({
                        'target': topic,
                        'relationship': f'Fundamental connection to {topic}',
                        'example': 'See detailed analysis in sources'
                    })
        except Exception as e:
            # Fallback on any parsing error
            relationships = [{
                'target': topic,
                'relationship': 'Related concept - see research',
                'example': 'Refer to authoritative sources'
            } for topic in connect_to]

        return relationships

    def _parse_derivation_steps(self, content: str) -> List[Dict[str, str]]:
        """Extract derivation steps from response."""
        steps = []

        try:
            # Extract numbered steps (1., 2., etc.)
            step_pattern = r'(?:step\s+)?(\d+)[.):\s]+(.+?)(?=(?:step\s+)?\d+[.):]|\Z)'
            matches = re.finditer(step_pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                step_text = (match.group(2) or "").strip()

                # Split into expression and explanation if possible
                expression = step_text[:200]
                explanation = ""
                justification = ""

                # Look for explanation markers
                if any(word in step_text.lower() for word in ['because', 'since', 'by', 'using']):
                    parts = re.split(r'\b(?:because|since|by|using)\b', step_text, 1, re.IGNORECASE)
                    if len(parts) == 2:
                        expression = (parts[0] or "").strip()[:200]
                        explanation = (parts[1] or "").strip()[:200]
                else:
                    explanation = step_text[:200]

                steps.append({
                    'expression': expression,
                    'explanation': explanation if explanation else 'Mathematical transformation',
                    'justification': justification if justification else 'Standard derivation rule'
                })

            # If no structured steps found, create one from content
            if not steps:
                steps = [{
                    'expression': content[:200] if content else 'See full derivation',
                    'explanation': 'Complete derivation available in response',
                    'justification': 'Refer to source material'
                }]

        except Exception as e:
            # Fallback on parsing error
            steps = [{
                'expression': 'Derivation step',
                'explanation': 'See research sources for complete derivation',
                'justification': 'Mathematical principles apply'
            }]

        return steps

    def _parse_problem_strategy(self, content: str) -> Dict[str, Any]:
        """Extract problem-solving strategy from response."""
        try:
            # Extract lists with safe fallbacks
            approach_steps = self._extract_list_items(content, "step.?by.?step|approach|methodology|steps")
            key_insights = self._extract_list_items(content, "key.?insights?|important.?points?|insights?")
            common_pitfalls = self._extract_list_items(content, "common.?pitfalls?|traps?|mistakes?|errors?")
            example_problem = self._extract_section(content, "example.?problem", multiline=True)

            return {
                'approach_steps': approach_steps if approach_steps else ['Identify problem type', 'Apply relevant technique', 'Verify solution'],
                'key_insights': key_insights if key_insights else ['Understand the underlying pattern', 'Practice similar problems'],
                'common_pitfalls': common_pitfalls if common_pitfalls else ['Rushing without understanding', 'Not checking work'],
                'example_problem': example_problem if example_problem else 'See sources for examples'
            }
        except Exception as e:
            # Safe fallback
            return {
                'approach_steps': ['Review problem carefully', 'Apply standard technique', 'Check solution'],
                'key_insights': ['Master fundamentals first', 'Practice regularly'],
                'common_pitfalls': ['Calculation errors', 'Misreading problem'],
                'example_problem': 'Refer to textbook examples'
            }

    def _parse_exam_pattern(self, content: str) -> Dict[str, Any]:
        """Extract exam pattern analysis from response."""
        try:
            # Extract high frequency topics
            high_frequency_topics = self._extract_list_items(content, "high.?frequency|common.?topics?|frequently.?tested")

            # Extract question types
            common_question_types = self._extract_list_items(content, "question.?types?|formats?|question.?formats?")

            # Extract study recommendations
            study_recommendations = self._extract_list_items(content, "study.?recommendations?|preparation.?tips?|study.?tips?")

            # Calculate confidence based on how much we extracted
            confidence = 0.7  # Base confidence
            if high_frequency_topics:
                confidence += 0.1
            if common_question_types:
                confidence += 0.1
            if study_recommendations:
                confidence += 0.1

            return {
                'high_frequency_topics': high_frequency_topics if high_frequency_topics else ['Core concepts', 'Problem solving', 'Theory application'],
                'common_question_types': common_question_types if common_question_types else ['Multiple choice', 'Short answer', 'Problem solving'],
                'study_recommendations': study_recommendations if study_recommendations else ['Review lecture notes', 'Practice problems', 'Study past exams'],
                'confidence': min(confidence, 1.0)
            }
        except Exception as e:
            # Safe fallback
            return {
                'high_frequency_topics': ['Review all course material', 'Focus on major concepts'],
                'common_question_types': ['Various question formats expected'],
                'study_recommendations': ['Study comprehensively', 'Practice regularly', 'Review all materials'],
                'confidence': 0.5
            }

    def _parse_notation(self, content: str) -> Dict[str, Any]:
        """Extract notation explanation from response."""
        try:
            name = self._extract_section(content, "name", multiline=False)
            meaning = self._extract_section(content, "meaning|represents?", multiline=True)

            # Extract contexts where this notation appears
            common_contexts = self._extract_list_items(content, "context|usage|used.?in|appears.?in|common.?in")

            # Extract example usage
            example_usage = self._extract_section(content, "examples?|example.?usage", multiline=True)

            return {
                'name': name if name else "Mathematical symbol",
                'meaning': meaning if meaning else "Mathematical notation - see sources for details",
                'common_contexts': common_contexts if common_contexts else ['Mathematics', 'Advanced calculus', 'Abstract algebra'],
                'example_usage': example_usage if example_usage else "Refer to authoritative sources for usage examples"
            }
        except Exception as e:
            # Safe fallback on any parsing error
            return {
                'name': 'Mathematical notation',
                'meaning': 'See research sources for complete explanation',
                'common_contexts': ['Mathematics', 'Science'],
                'example_usage': 'Consult textbooks and authoritative sources'
            }

    def _extract_section(self, content: str, pattern: str, multiline: bool = True) -> str:
        """Extract a section from content using regex pattern."""
        flags = re.IGNORECASE | (re.DOTALL if multiline else 0)
        match = re.search(f"{pattern}[:\\s]*(.+?)(?=\\n\\n|\\n[A-Z]+:|$)", content, flags)
        return (match.group(1) or "").strip() if match else ""

    def _extract_list_items(self, content: str, section_pattern: str) -> List[str]:
        """Extract bulleted/numbered list items from a section."""
        section = self._extract_section(content, section_pattern, multiline=True)
        if not section:
            return []

        # Match bullet points, numbered lists, or dashes
        items = re.findall(r'(?:^|\n)\s*(?:[•\-*]|\d+[.)])\s*(.+)', section, re.MULTILINE)
        return [item.strip() for item in items if item and len(item.strip()) > 5][:10]  # Max 10 items

    def _extract_importance(self, content: str) -> str:
        """Extract overall importance statement."""
        importance = self._extract_section(content, "importance|why.?this.?matters", multiline=True)
        return importance or "Understanding these connections strengthens conceptual foundation"

    def _parse_visual_intuition(self, content: str) -> Dict[str, Any]:
        """Extract visual intuition data from response."""
        try:
            analogies = self._extract_list_items(content, "analog(?:y|ies)")
            mental_models = self._extract_list_items(content, "mental.?model")
            learning_approach = self._extract_section(content, "learning.?approach|strategy", multiline=True)

            # Extract visual resources
            visual_resources = []
            video_items = self._extract_list_items(content, "video|animation")
            diagram_items = self._extract_list_items(content, "diagram|visual|graph")

            for item in video_items[:3]:
                visual_resources.append({'type': 'video', 'description': item, 'source': 'See sources'})
            for item in diagram_items[:3]:
                visual_resources.append({'type': 'diagram', 'description': item, 'source': 'See sources'})

            return {
                'analogies': analogies if analogies else ['Refer to sources for analogies'],
                'mental_models': mental_models if mental_models else ['Refer to sources for mental models'],
                'visual_resources': visual_resources if visual_resources else [{'type': 'diagram', 'description': 'See research sources', 'source': ''}],
                'learning_approach': learning_approach if learning_approach else 'Study fundamentals and practice regularly'
            }
        except Exception as e:
            # Safe fallback on any parsing error
            return {
                'analogies': ['See research sources for analogies'],
                'mental_models': ['See research sources for mental models'],
                'visual_resources': [{'type': 'reference', 'description': 'Consult educational resources', 'source': ''}],
                'learning_approach': 'Study systematically with practice'
            }

    def _parse_formula_sheet(self, content: str) -> Dict[str, Any]:
        """Extract formula sheet data from response."""
        try:
            key_relationships = self._extract_list_items(content, "relationship|connect")
            common_mistakes = self._extract_list_items(content, "mistake|error|pitfall")

            # Extract categories with formulas
            categories = []
            category_patterns = [
                "derivative", "integral", "trigonometry", "logarithm", "exponential",
                "algebra", "geometry", "calculus", "statistics", "probability"
            ]

            for category_name in category_patterns:
                formulas = self._extract_list_items(content, category_name)
                if formulas:
                    categories.append({
                        'name': category_name.title(),
                        'formulas': formulas
                    })

            # If no categorized formulas found, extract all formula-like items
            if not categories:
                all_formulas = self._extract_list_items(content, "formula|equation")
                if all_formulas:
                    categories.append({
                        'name': 'Key Formulas',
                        'formulas': all_formulas
                    })

            return {
                'categories': categories if categories else [{'name': 'Formulas', 'formulas': ['Refer to sources for formulas']}],
                'key_relationships': key_relationships if key_relationships else ['Refer to sources for relationships'],
                'common_mistakes': common_mistakes if common_mistakes else ['Review fundamentals carefully', 'Practice regularly']
            }
        except Exception as e:
            # Safe fallback
            return {
                'categories': [{'name': 'Formulas', 'formulas': ['Consult textbooks for formulas']}],
                'key_relationships': ['See research sources for relationships'],
                'common_mistakes': ['Practice with care', 'Review solutions']
            }

    def _parse_paper_summary(self, content: str) -> Dict[str, Any]:
        """Extract paper summary data from response."""
        try:
            main_contribution = self._extract_section(content, "main.?contribution|contribution", multiline=True)
            key_findings = self._extract_list_items(content, "key.?finding|finding|result")
            methodology = self._extract_section(content, "methodology|method|approach", multiline=True)
            limitations = self._extract_list_items(content, "limitation|future.?work")

            # Extract authors if not provided
            authors = self._extract_section(content, "author", multiline=False)

            # Extract year if not provided
            year = self._extract_section(content, "year|\\d{4}", multiline=False)

            # Extract relevance score (look for numbers 0-1 or percentages)
            import re
            relevance_match = re.search(r'(?:relevance|significance|impact).*?(\d+\.?\d*)', content, re.IGNORECASE)
            if relevance_match:
                score = float(relevance_match.group(1))
                relevance_score = score if score <= 1.0 else score / 100.0
            else:
                relevance_score = 0.7  # Default moderate relevance

            return {
                'authors': authors if authors else 'Unknown',
                'year': year if year else '',
                'main_contribution': main_contribution if main_contribution else 'See sources for contribution details',
                'key_findings': key_findings if key_findings else ['Refer to sources for findings'],
                'methodology': methodology if methodology else 'See sources for methodology',
                'limitations': limitations if limitations else ['Refer to paper for limitations'],
                'relevance_score': min(max(relevance_score, 0.0), 1.0)  # Clamp between 0 and 1
            }
        except Exception as e:
            # Safe fallback
            return {
                'authors': 'Unknown',
                'year': '',
                'main_contribution': 'Consult paper for details',
                'key_findings': ['See research sources'],
                'methodology': 'See research sources',
                'limitations': ['Refer to original paper'],
                'relevance_score': 0.5
            }
