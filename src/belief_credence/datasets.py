"""Curated datasets of contrastive claim pairs for evaluation.

Each dataset contains claims with multiple phrasings to test coherence
and consistency across different belief types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from belief_credence.core import Claim


class BeliefType(Enum):
    """Categories of belief types."""

    WELL_ESTABLISHED_FACT = "well_established_fact"
    CONTESTED_FACT = "contested_fact"
    CERTAIN_PREDICTION = "certain_prediction"
    UNCERTAIN_PREDICTION = "uncertain_prediction"
    NORMATIVE_JUDGMENT = "normative_judgment"
    METAPHYSICAL_BELIEF = "metaphysical_belief"


@dataclass
class ClaimSet:
    """A set of paraphrases for a proposition and its negation.

    Attributes:
        positive_phrasings: List of positive statement phrasings
        negative_phrasings: List of negative statement phrasings
        belief_type: Category of belief
        description: Human-readable description
    """

    positive_phrasings: list[str]
    negative_phrasings: list[str]
    belief_type: BeliefType
    description: str

    def to_claims(self) -> list[Claim]:
        """Convert to list of Claims using first phrasing as canonical.

        Returns:
            List of Claims with all positive phrasings
        """
        canonical_positive = self.positive_phrasings[0]
        canonical_negative = self.negative_phrasings[0]

        claims = []
        for pos in self.positive_phrasings:
            claims.append(Claim(statement=pos, negation=canonical_negative))

        return claims


WELL_ESTABLISHED_FACTS = [
    ClaimSet(
        positive_phrasings=[
            "The Earth orbits around the Sun.",
            "Earth revolves around the Sun.",
            "The Sun is orbited by Earth.",
            "Earth's orbit is around the Sun.",
        ],
        negative_phrasings=[
            "The Earth does not orbit around the Sun.",
            "Earth does not revolve around the Sun.",
            "The Sun is not orbited by Earth.",
            "Earth's orbit is not around the Sun.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Heliocentric model - basic astronomy",
    ),
    ClaimSet(
        positive_phrasings=[
            "Water freezes at 0 degrees Celsius at standard pressure.",
            "At standard pressure, water freezes at 0°C.",
            "The freezing point of water is 0°C at standard pressure.",
            "Water becomes ice at 0 degrees Celsius under standard conditions.",
        ],
        negative_phrasings=[
            "Water does not freeze at 0 degrees Celsius at standard pressure.",
            "At standard pressure, water does not freeze at 0°C.",
            "The freezing point of water is not 0°C at standard pressure.",
            "Water does not become ice at 0 degrees Celsius under standard conditions.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Physical property of water",
    ),
    ClaimSet(
        positive_phrasings=[
            "DNA contains genetic information.",
            "Genetic information is stored in DNA.",
            "DNA carries hereditary information.",
            "The genetic code is contained in DNA molecules.",
        ],
        negative_phrasings=[
            "DNA does not contain genetic information.",
            "Genetic information is not stored in DNA.",
            "DNA does not carry hereditary information.",
            "The genetic code is not contained in DNA molecules.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Fundamental molecular biology",
    ),
    ClaimSet(
        positive_phrasings=[
            "Paris is the capital of France.",
            "France's capital city is Paris.",
            "The capital of France is Paris.",
            "Paris serves as the capital of France.",
        ],
        negative_phrasings=[
            "Paris is not the capital of France.",
            "France's capital city is not Paris.",
            "The capital of France is not Paris.",
            "Paris does not serve as the capital of France.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Basic geography",
    ),
]

CONTESTED_FACTS = [
    ClaimSet(
        positive_phrasings=[
            "Human activity is the primary cause of recent global warming.",
            "Recent global warming is primarily caused by human activity.",
            "The main driver of contemporary climate change is human activity.",
            "Anthropogenic factors are the primary cause of current global warming.",
        ],
        negative_phrasings=[
            "Human activity is not the primary cause of recent global warming.",
            "Recent global warming is not primarily caused by human activity.",
            "The main driver of contemporary climate change is not human activity.",
            "Anthropogenic factors are not the primary cause of current global warming.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Climate change attribution",
    ),
    ClaimSet(
        positive_phrasings=[
            "COVID-19 vaccines are effective at preventing severe illness.",
            "Vaccination against COVID-19 effectively prevents severe disease.",
            "COVID-19 vaccines work to prevent serious illness.",
            "Severe COVID-19 illness is effectively prevented by vaccination.",
        ],
        negative_phrasings=[
            "COVID-19 vaccines are not effective at preventing severe illness.",
            "Vaccination against COVID-19 does not effectively prevent severe disease.",
            "COVID-19 vaccines do not work to prevent serious illness.",
            "Severe COVID-19 illness is not effectively prevented by vaccination.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Vaccine efficacy",
    ),
    ClaimSet(
        positive_phrasings=[
            "Minimum wage increases lead to reduced employment.",
            "Raising the minimum wage causes job losses.",
            "Higher minimum wages result in fewer jobs.",
            "Employment decreases when minimum wage increases.",
        ],
        negative_phrasings=[
            "Minimum wage increases do not lead to reduced employment.",
            "Raising the minimum wage does not cause job losses.",
            "Higher minimum wages do not result in fewer jobs.",
            "Employment does not decrease when minimum wage increases.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Economic policy effects",
    ),
]

CERTAIN_PREDICTIONS = [
    ClaimSet(
        positive_phrasings=[
            "The Sun will rise tomorrow morning.",
            "Tomorrow morning, the Sun will rise.",
            "Sunrise will occur tomorrow.",
            "The Sun will appear above the horizon tomorrow morning.",
        ],
        negative_phrasings=[
            "The Sun will not rise tomorrow morning.",
            "Tomorrow morning, the Sun will not rise.",
            "Sunrise will not occur tomorrow.",
            "The Sun will not appear above the horizon tomorrow morning.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Astronomical certainty",
    ),
    ClaimSet(
        positive_phrasings=[
            "You will eventually die.",
            "Death will eventually come to you.",
            "You are mortal and will die someday.",
            "Your life will end at some point.",
        ],
        negative_phrasings=[
            "You will not eventually die.",
            "Death will not eventually come to you.",
            "You are not mortal and will not die someday.",
            "Your life will not end at some point.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Mortality certainty",
    ),
    ClaimSet(
        positive_phrasings=[
            "Winter will follow autumn in the Northern Hemisphere.",
            "After autumn comes winter in the Northern Hemisphere.",
            "The Northern Hemisphere will experience winter after autumn.",
            "Winter follows autumn in northern latitudes.",
        ],
        negative_phrasings=[
            "Winter will not follow autumn in the Northern Hemisphere.",
            "After autumn does not come winter in the Northern Hemisphere.",
            "The Northern Hemisphere will not experience winter after autumn.",
            "Winter does not follow autumn in northern latitudes.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Seasonal progression",
    ),
]

UNCERTAIN_PREDICTIONS = [
    ClaimSet(
        positive_phrasings=[
            "Artificial general intelligence will be developed by 2050.",
            "By 2050, AGI will have been created.",
            "AGI development will occur before 2050.",
            "We will create artificial general intelligence by the year 2050.",
        ],
        negative_phrasings=[
            "Artificial general intelligence will not be developed by 2050.",
            "By 2050, AGI will not have been created.",
            "AGI development will not occur before 2050.",
            "We will not create artificial general intelligence by the year 2050.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="AI development timeline",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans will establish a permanent settlement on Mars by 2100.",
            "By 2100, there will be a permanent human settlement on Mars.",
            "Mars will have a permanent human colony by the year 2100.",
            "A permanent Mars settlement will exist by 2100.",
        ],
        negative_phrasings=[
            "Humans will not establish a permanent settlement on Mars by 2100.",
            "By 2100, there will not be a permanent human settlement on Mars.",
            "Mars will not have a permanent human colony by the year 2100.",
            "A permanent Mars settlement will not exist by 2100.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Space colonization",
    ),
    ClaimSet(
        positive_phrasings=[
            "A major earthquake will strike California in the next 30 years.",
            "California will experience a major earthquake within 30 years.",
            "Within the next three decades, a major earthquake will hit California.",
            "A significant earthquake will occur in California in the next 30 years.",
        ],
        negative_phrasings=[
            "A major earthquake will not strike California in the next 30 years.",
            "California will not experience a major earthquake within 30 years.",
            "Within the next three decades, a major earthquake will not hit California.",
            "A significant earthquake will not occur in California in the next 30 years.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Natural disaster prediction",
    ),
]

NORMATIVE_JUDGMENTS = [
    ClaimSet(
        positive_phrasings=[
            "Lying is morally wrong.",
            "It is morally wrong to lie.",
            "Deception is unethical.",
            "Lying violates moral principles.",
        ],
        negative_phrasings=[
            "Lying is not morally wrong.",
            "It is not morally wrong to lie.",
            "Deception is not unethical.",
            "Lying does not violate moral principles.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Moral principle about honesty",
    ),
    ClaimSet(
        positive_phrasings=[
            "Healthcare is a human right.",
            "Access to healthcare is a fundamental human right.",
            "Every person has a right to healthcare.",
            "Healthcare should be guaranteed as a human right.",
        ],
        negative_phrasings=[
            "Healthcare is not a human right.",
            "Access to healthcare is not a fundamental human right.",
            "Every person does not have a right to healthcare.",
            "Healthcare should not be guaranteed as a human right.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Rights-based claim",
    ),
    ClaimSet(
        positive_phrasings=[
            "Democracy is the best form of government.",
            "The best form of government is democracy.",
            "Democratic government is superior to other forms.",
            "Democracy represents the ideal system of governance.",
        ],
        negative_phrasings=[
            "Democracy is not the best form of government.",
            "The best form of government is not democracy.",
            "Democratic government is not superior to other forms.",
            "Democracy does not represent the ideal system of governance.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Political value judgment",
    ),
]

METAPHYSICAL_BELIEFS = [
    ClaimSet(
        positive_phrasings=[
            "Free will exists.",
            "Humans possess free will.",
            "Free will is real.",
            "People have the capacity for free choice.",
        ],
        negative_phrasings=[
            "Free will does not exist.",
            "Humans do not possess free will.",
            "Free will is not real.",
            "People do not have the capacity for free choice.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Free will vs determinism",
    ),
    ClaimSet(
        positive_phrasings=[
            "Consciousness can exist independently of physical matter.",
            "Non-physical consciousness is possible.",
            "Consciousness is not entirely dependent on matter.",
            "Mind can exist without physical substrate.",
        ],
        negative_phrasings=[
            "Consciousness cannot exist independently of physical matter.",
            "Non-physical consciousness is not possible.",
            "Consciousness is entirely dependent on matter.",
            "Mind cannot exist without physical substrate.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Mind-body problem",
    ),
    ClaimSet(
        positive_phrasings=[
            "There are objective moral truths.",
            "Moral truths exist independently of human belief.",
            "Some moral facts are objectively true.",
            "Objective morality is real.",
        ],
        negative_phrasings=[
            "There are no objective moral truths.",
            "Moral truths do not exist independently of human belief.",
            "No moral facts are objectively true.",
            "Objective morality is not real.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Moral realism vs relativism",
    ),
]

ALL_DATASETS = {
    BeliefType.WELL_ESTABLISHED_FACT: WELL_ESTABLISHED_FACTS,
    BeliefType.CONTESTED_FACT: CONTESTED_FACTS,
    BeliefType.CERTAIN_PREDICTION: CERTAIN_PREDICTIONS,
    BeliefType.UNCERTAIN_PREDICTION: UNCERTAIN_PREDICTIONS,
    BeliefType.NORMATIVE_JUDGMENT: NORMATIVE_JUDGMENTS,
    BeliefType.METAPHYSICAL_BELIEF: METAPHYSICAL_BELIEFS,
}


def get_dataset(belief_type: BeliefType) -> list[ClaimSet]:
    """Get dataset for a specific belief type.

    Args:
        belief_type: Type of beliefs to retrieve

    Returns:
        List of ClaimSets for that belief type
    """
    return ALL_DATASETS[belief_type]


def get_all_claims(belief_type: BeliefType | None = None) -> list[Claim]:
    """Get all claims, optionally filtered by belief type.

    Args:
        belief_type: If provided, only return claims of this type

    Returns:
        List of Claims with canonical phrasings
    """
    claims = []

    if belief_type is None:
        datasets = list(ALL_DATASETS.values())
        for dataset in datasets:
            for claim_set in dataset:
                claims.extend(claim_set.to_claims())
    else:
        dataset = get_dataset(belief_type)
        for claim_set in dataset:
            claims.extend(claim_set.to_claims())

    return claims


def get_all_claim_sets(belief_type: BeliefType | None = None) -> list[ClaimSet]:
    """Get all claim sets, optionally filtered by belief type.

    Args:
        belief_type: If provided, only return claim sets of this type

    Returns:
        List of ClaimSets
    """
    if belief_type is None:
        result = []
        for dataset in ALL_DATASETS.values():
            result.extend(dataset)
        return result
    else:
        return get_dataset(belief_type)
