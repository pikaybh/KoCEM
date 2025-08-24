from typing import Dict, List, Literal, ClassVar

import datasets
from pydantic import BaseModel, Field


DifficultyType = Literal["Easy", "Medium", "Hard"]
SplitType = Literal["dev", "extra", "test", "val"]
LocaleType = Literal["en", "ko", "zh", "ja", "es", "fr", "de", "it", "pt", "ru"]
DimensionType = Literal[
    "Drawing Interpretation",
    "Expert-level Reasoning",
    "Domain-specific Terminology",
    "Domain-specific Knowledge",
    "Exper-level Comprehension"
]



class Subject(BaseModel):
    """
    Subject model for defining the subject of a question.
    """
    abbreviation: str = Field(..., description="Abbreviation of the subject")
    description: str = Field(..., description="Description of the subject")
    dimension: DimensionType = Field(..., description="Dimension of the subject")
    split: Dict[SplitType, int | None] = Field(..., description="List of splits the subject belongs to")
    locale: List[LocaleType] = Field(..., description="Locale of the subject")
    has_difficulty: bool = Field(False, description="Indicates if the subject has difficulty levels")



class KoCEM(BaseModel):
    name: str = "pikaybh/KoCEM"
    Architectural_Planning: ClassVar[Subject] = Subject(
        abbreviation="ap",
        description="Architectural Planning",
        dimension="Domain-specific Knowledge",
        split={"dev": 3, "test": 461, "val": 41}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Building_System: ClassVar[Subject] = Subject(
        abbreviation="bs",
        description="Building System",
        dimension="Domain-specific Knowledge",
        split={"dev": 5, "test": 367, "val": 49}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Comprehensive_Understanding: ClassVar[Subject] = Subject(
        abbreviation="cu",
        description="Comprehensive Understanding",
        dimension="Exper-level Comprehension",
        split={"dev": 3, "test": 161, "val": 157}, 
        locale=["en", "ko"],
        has_difficulty=False
    )
    Construction_Management: ClassVar[Subject] = Subject(
        abbreviation="cm",
        description="Construction Management",
        dimension="Domain-specific Knowledge",
        split={"dev": 5, "test": 488, "val": 34}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Domain_Reasoning: ClassVar[Subject] = Subject(
        abbreviation="dr",
        description="Domain Reasoning",
        dimension="Expert-level Reasoning",
        split={"dev": 3, "test": 255, "val": 10}, 
        locale=["en", "ko"],
        has_difficulty=False
    )
    Drawing_Interpretation: ClassVar[Subject] = Subject(
        abbreviation="di",
        description="Drawing Interpretation",
        dimension="Drawing Interpretation",
        split={"dev": 3, "test": 122, "val": 9}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Industry_Jargon: ClassVar[Subject] = Subject(
        abbreviation="ij",
        description="Industry Jargon",
        dimension="Domain-specific Terminology",
        split={"dev": None, "test": None, "val": None}, 
        locale=["en", "ko"]
    )
    Interior: ClassVar[Subject] = Subject(
        abbreviation="int",
        description="Interior",
        dimension="Domain-specific Knowledge",
        split={"dev": 6, "test": 357, "val": 46}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Materials: ClassVar[Subject] = Subject(
        abbreviation="mat",
        description="Materials",
        dimension="Domain-specific Knowledge",
        split={"dev": 8, "test": 407, "val": 43}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Safety_Management: ClassVar[Subject] = Subject(
        abbreviation="sm",
        description="Safety Management",
        dimension="Domain-specific Knowledge",
        split={"dev": 4, "test": 376, "val": 41}, 
        locale=["en", "ko"],
        has_difficulty=True
    )
    Standard_Nomenclature: ClassVar[Subject] = Subject(
        abbreviation="sn",
        description="Standard Nomenclature",
        dimension="Domain-specific Terminology",
        split={"dev": 5, "test": 450, "val": 45}, 
        locale=["en", "ko"],
        has_difficulty=False
    )
    Structural_Engineering: ClassVar[Subject] = Subject(
        abbreviation="se",
        description="Structural Engineering",
        dimension="Domain-specific Knowledge",
        split={"dev": 2, "test": 342, "val": 34}, 
        locale=["en", "ko"],
        has_difficulty=True
    )

    @classmethod
    def list_subjects(cls) -> List[Subject]:
        """Return all subject definitions."""
        return [
            cls.Architectural_Planning,
            cls.Building_System,
            cls.Construction_Management,
            cls.Comprehensive_Understanding,
            cls.Drawing_Interpretation,
            cls.Domain_Reasoning,
            cls.Industry_Jargon,
            cls.Interior,
            cls.Materials,
            cls.Safety_Management,
            cls.Structural_Engineering,
            cls.Standard_Nomenclature,
        ]



# Define full schema to avoid Arrow inferring null types on empty/problematic splits
KOCEM_FEATURES = datasets.Features({
    "id": datasets.Value("string"),
    "question": datasets.Value("string"),
    "options": datasets.Value("string"),
    "answer": datasets.Value("string"),
    "explanation": datasets.Value("string"),
    "image": {
        "bytes": datasets.Value("binary"),
        "path": datasets.Value("string"),
    },
    "question_type": datasets.Value("string"),
    "field": datasets.Value("string"),
    "subfield": datasets.Value("string"),
    "korean_national_technical_certification": datasets.Value("string"),
    "exam": datasets.Value("string"),
    "date": datasets.Value("string"),
    "subject": datasets.Value("string"),
    "human_acc": datasets.Value("float64"),
    "difficulty": datasets.Value("string"),
    "answer_key": datasets.Value("string"),
    "ko_question": datasets.Value("string"),
    "en_question": datasets.Value("string"),
    "ko_options": datasets.Value("string"),
    "en_options": datasets.Value("string"),
    "ko_answer": datasets.Value("string"),
    "en_answer": datasets.Value("string"),
    "ko_explanation": datasets.Value("string"),
    "en_explanation": datasets.Value("string"),
    "eval": datasets.Value("string"),
    "eval_loop": datasets.Value("string"),
    "human_feedback": datasets.Value("int64"),
    "field_feedback": datasets.Value("string"),
})
DOMAIN_RESONING_FEATURES = datasets.Features({
    'image': {
        'bytes': datasets.Value('binary'), 
        'path': datasets.Value('string')
    },
    'date': datasets.Value('string'),
    'number': datasets.Value('string'),
    'question': datasets.Value('string'),
    'explanation': datasets.Value('string'),
    'answer': datasets.Value('string'),
    'id': datasets.Value('string'),
    'answer_key': datasets.Value('string'),
    'options': datasets.Value('string'),
    'question_type': datasets.Value('string'),
    'field': datasets.Value('string'),
    'korean_national_technical_certification': datasets.Value('string'),
    'exam': datasets.Value('string'),
    'subject': datasets.Value('string'),
    'human_acc': datasets.Value('string'),
    'difficulty': datasets.Value('string'),
    'subfield': datasets.Value('string'),
    'ko_question': datasets.Value('string'),
    'en_question': datasets.Value('string'),
    'ko_options': datasets.Value('string'),
    'en_options': datasets.Value('string'),
    'ko_answer': datasets.Value('string'),
    'en_answer': datasets.Value('string'),
    'ko_explanation': datasets.Value('string'),
    'en_explanation': datasets.Value('string'),
    'eval': datasets.Value('string'),
    'eval_loop': datasets.Value('string'),
    'human_feedback': datasets.Value('int64'),
    'field_feedback': datasets.Value('string'),
})
DRAWING_INTERPRETATION_FEATURES = datasets.Features({
    "image": {
        "bytes": datasets.Value("binary"), 
        "path": datasets.Value("string")
    },
    "date": datasets.Value("int64"),          # 원본에 맞춤
    "number": datasets.Value("int64"),        # 원본에 맞춤 (임시 보유)
    "subfield": datasets.Value("string"),
    "question": datasets.Value("string"),
    "subject": datasets.Value("string"),
    "options": datasets.Value("string"),
    "answer": datasets.Value("string"),
    "answer_key": datasets.Value("string"),
    "explanation": datasets.Value("string"),
    "question_type": datasets.Value("string"),
    "field": datasets.Value("string"),
    "korean_national_technical_certification": datasets.Value("string"),
    "exam": datasets.Value("string"),
    "human_acc": datasets.Value("string"),    # 원본에 맞춤
    "difficulty": datasets.Value("string"),
    "id": datasets.Value("string"),
    "ko_question": datasets.Value("string"),
    "en_question": datasets.Value("string"),
    "ko_options": datasets.Value("string"),
    "en_options": datasets.Value("string"),
    "ko_answer": datasets.Value("string"),
    "en_answer": datasets.Value("string"),
    "ko_explanation": datasets.Value("string"),
    "en_explanation": datasets.Value("string"),
    "eval": datasets.Value("string"),
    "eval_loop": datasets.Value("string"),
    "human_feedback": datasets.Value("int64"),
    "field_feedback": datasets.Value("string"),
})
STANDARD_NOMENCLATURE_FEATURES = datasets.Features({
    "question": datasets.Value("string"),
    "options": datasets.Sequence(
        datasets.Value("string")
    ),
    "answer": datasets.Value("string"),
    "answer_key": datasets.Value("string"),
    "explanation": datasets.Value("string"),
    "question_type": datasets.Value("string"),
    "field": datasets.Value("string"),
    "subfield": datasets.Value("string"),
    "korean_national_technical_certification": datasets.Value("string"),
    "exam": datasets.Value("string"),
    "subject": datasets.Value("string"),
    "human_acc": datasets.Value("string"),    # 원본에 맞춤
    "difficulty": datasets.Value("string"),
    "image": {
        "bytes": datasets.Value("binary"), 
        "path": datasets.Value("string")
    },
    "eval_loop": datasets.Value("string"),
    "id": datasets.Value("string")
})

__all__ = [
    'DifficultyType',
    'DimensionType', 
    'LocaleType', 
    'SplitType', 
    'Subject', 
    'KoCEM', 
    'KOCEM_FEATURES',
    'DOMAIN_RESONING_FEATURES',
    'DRAWING_INTERPRETATION_FEATURES',
    'STANDARD_NOMENCLATURE_FEATURES'
]