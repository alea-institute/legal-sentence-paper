"""
Standard legal sentence examples with expected boundaries.

This module provides standardized legal text examples with explicitly
defined expected sentence boundaries for testing tokenizers.
"""

from typing import Dict, List, Any, Tuple

# Define legal sentences with expected boundaries
# Format: Each example has text and expected_sentences (list of sentences)

LEGAL_SENTENCES = [
    {
        "name": "Basic two sentences",
        "text": "This is a simple first sentence. This is the second sentence.",
        "expected_sentences": [
            "This is a simple first sentence.",
            "This is the second sentence."
        ]
    },
    {
        "name": "Legal case citation",
        "text": "The Court applied Smith v. Jones, 123 U.S. 456 (1950). This precedent is binding.",
        "expected_sentences": [
            "The Court applied Smith v. Jones, 123 U.S. 456 (1950).",
            "This precedent is binding."
        ]
    },
    {
        "name": "Legal statute citation",
        "text": "Under 18 U.S.C. § 1343, wire fraud is a federal offense. The punishment is severe.",
        "expected_sentences": [
            "Under 18 U.S.C. § 1343, wire fraud is a federal offense.",
            "The punishment is severe."
        ]
    },
    {
        "name": "Abbreviations",
        "text": "The plaintiff, Dr. Smith, filed against Corp. Inc. The court ruled in favor of the plaintiff.",
        "expected_sentences": [
            "The plaintiff, Dr. Smith, filed against Corp. Inc.",
            "The court ruled in favor of the plaintiff."
        ]
    },
    {
        "name": "Quotations with attribution",
        "text": "The court stated: \"The evidence is insufficient.\" This ruling was appealed.",
        "expected_sentences": [
            "The court stated: \"The evidence is insufficient.\"",
            "This ruling was appealed."
        ]
    },
    {
        "name": "Case name with periods",
        "text": "In U.S. v. Johnson, the Court established a new test. This test has three parts.",
        "expected_sentences": [
            "In U.S. v. Johnson, the Court established a new test.",
            "This test has three parts."
        ]
    },
    {
        "name": "Numbered list items",
        "text": "The Court identified three factors: 1. Timing of disclosure. 2. Prejudice to the defendant. 3. Evidence of bad faith.",
        "expected_sentences": [
            "The Court identified three factors:",
            "1. Timing of disclosure.",
            "2. Prejudice to the defendant.",
            "3. Evidence of bad faith."
        ]
    },
    {
        "name": "Parenthetical with period",
        "text": "The evidence (including the testimony of Dr. Jones) was compelling. The jury agreed.",
        "expected_sentences": [
            "The evidence (including the testimony of Dr. Jones) was compelling.",
            "The jury agreed."
        ]
    },
    {
        "name": "Section references",
        "text": "The contract violates Section 2.3.4. This makes it void.",
        "expected_sentences": [
            "The contract violates Section 2.3.4.",
            "This makes it void."
        ]
    },
    {
        "name": "Multiple citations",
        "text": "See Smith v. Jones, 123 F.3d 456 (7th Cir. 1996); Johnson v. Williams, 789 F.2d 012 (2d Cir. 1985). These cases establish the doctrine.",
        "expected_sentences": [
            "See Smith v. Jones, 123 F.3d 456 (7th Cir. 1996); Johnson v. Williams, 789 F.2d 012 (2d Cir. 1985).",
            "These cases establish the doctrine."
        ]
    },
    {
        "name": "Dated document",
        "text": "The agreement is dated January 1, 2023. It expires after one year.",
        "expected_sentences": [
            "The agreement is dated January 1, 2023.",
            "It expires after one year."
        ]
    },
    {
        "name": "Quote with question",
        "text": "The court asked: \"What evidence supports this claim?\" No answer was provided.",
        "expected_sentences": [
            "The court asked: \"What evidence supports this claim?\"",
            "No answer was provided."
        ]
    },
    {
        "name": "Semicolon",
        "text": "The plaintiff presented evidence; the defendant objected. The judge overruled.",
        "expected_sentences": [
            "The plaintiff presented evidence; the defendant objected.",
            "The judge overruled."
        ]
    },
    {
        "name": "Case name with multiple dots",
        "text": "In F.T.C. v. Corp., Inc., the Court ruled on jurisdiction. Venue was proper.",
        "expected_sentences": [
            "In F.T.C. v. Corp., Inc., the Court ruled on jurisdiction.",
            "Venue was proper."
        ]
    },
    {
        "name": "Legal term with internal period",
        "text": "The e.g. evidence was excluded. The i.e. testimony was admitted.",
        "expected_sentences": [
            "The e.g. evidence was excluded.",
            "The i.e. testimony was admitted."
        ]
    }
]


def get_example_by_name(name: str) -> Dict[str, Any]:
    """Get a legal example by name.
    
    Args:
        name: Name of the example to retrieve
        
    Returns:
        Example dictionary or None if not found
    """
    for example in LEGAL_SENTENCES:
        if example["name"] == name:
            return example
    return None


def get_all_examples() -> List[Dict[str, Any]]:
    """Get all legal examples.
    
    Returns:
        List of all example dictionaries
    """
    return LEGAL_SENTENCES.copy()


def get_example_names() -> List[str]:
    """Get names of all available examples.
    
    Returns:
        List of example names
    """
    return [example["name"] for example in LEGAL_SENTENCES]


def get_simple_examples() -> List[Dict[str, Any]]:
    """Get a subset of simple examples without complex legal terminology.
    
    Returns:
        List of simple example dictionaries
    """
    simple_names = [
        "Basic two sentences",
        "Dated document",
        "Quotations with attribution",
        "Parenthetical with period",
        "Semicolon"
    ]
    
    return [ex for ex in LEGAL_SENTENCES if ex["name"] in simple_names]


def get_complex_examples() -> List[Dict[str, Any]]:
    """Get a subset of complex examples with legal terminology.
    
    Returns:
        List of complex example dictionaries
    """
    simple_names = [name for ex in get_simple_examples() for name in [ex["name"]]]
    return [ex for ex in LEGAL_SENTENCES if ex["name"] not in simple_names]