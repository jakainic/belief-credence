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
        ],
        negative_phrasings=[
            "The Earth does not orbit around the Sun.",
            "Earth does not revolve around the Sun.",
            "The Sun is not orbited by Earth.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Heliocentric model",
    ),
    ClaimSet(
        positive_phrasings=[
            "Water freezes at 0 degrees Celsius at standard pressure.",
            "At standard pressure, water freezes at 0°C.",
            "The freezing point of water is 0°C.",
        ],
        negative_phrasings=[
            "Water does not freeze at 0 degrees Celsius at standard pressure.",
            "At standard pressure, water does not freeze at 0°C.",
            "The freezing point of water is not 0°C.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Water freezing point",
    ),
    ClaimSet(
        positive_phrasings=[
            "DNA contains genetic information.",
            "Genetic information is stored in DNA.",
            "DNA carries hereditary information.",
        ],
        negative_phrasings=[
            "DNA does not contain genetic information.",
            "Genetic information is not stored in DNA.",
            "DNA does not carry hereditary information.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="DNA and genetics",
    ),
    ClaimSet(
        positive_phrasings=[
            "Paris is the capital of France.",
            "France's capital city is Paris.",
            "The capital of France is Paris.",
        ],
        negative_phrasings=[
            "Paris is not the capital of France.",
            "France's capital city is not Paris.",
            "The capital of France is not Paris.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="French capital",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans need oxygen to survive.",
            "Oxygen is necessary for human survival.",
            "Human survival requires oxygen.",
        ],
        negative_phrasings=[
            "Humans do not need oxygen to survive.",
            "Oxygen is not necessary for human survival.",
            "Human survival does not require oxygen.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Oxygen requirement",
    ),
    ClaimSet(
        positive_phrasings=[
            "Light travels faster than sound.",
            "Sound travels slower than light.",
            "The speed of light exceeds the speed of sound.",
        ],
        negative_phrasings=[
            "Light does not travel faster than sound.",
            "Sound does not travel slower than light.",
            "The speed of light does not exceed the speed of sound.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Light vs sound speed",
    ),
    ClaimSet(
        positive_phrasings=[
            "The Earth is approximately spherical.",
            "Earth has a roughly spherical shape.",
            "The shape of Earth is essentially spherical.",
        ],
        negative_phrasings=[
            "The Earth is not approximately spherical.",
            "Earth does not have a roughly spherical shape.",
            "The shape of Earth is not essentially spherical.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Earth's shape",
    ),
    ClaimSet(
        positive_phrasings=[
            "Gravity causes objects to fall toward Earth.",
            "Objects fall toward Earth due to gravity.",
            "Gravitational force pulls objects toward Earth.",
        ],
        negative_phrasings=[
            "Gravity does not cause objects to fall toward Earth.",
            "Objects do not fall toward Earth due to gravity.",
            "Gravitational force does not pull objects toward Earth.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Gravity and falling",
    ),
    ClaimSet(
        positive_phrasings=[
            "The speed of light in vacuum is constant.",
            "Light speed in vacuum remains constant.",
            "In a vacuum, light always travels at the same speed.",
        ],
        negative_phrasings=[
            "The speed of light in vacuum is not constant.",
            "Light speed in vacuum does not remain constant.",
            "In a vacuum, light does not always travel at the same speed.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Constant light speed",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans have 46 chromosomes.",
            "The human genome contains 46 chromosomes.",
            "There are 46 chromosomes in human cells.",
        ],
        negative_phrasings=[
            "Humans do not have 46 chromosomes.",
            "The human genome does not contain 46 chromosomes.",
            "There are not 46 chromosomes in human cells.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Human chromosomes",
    ),
    ClaimSet(
        positive_phrasings=[
            "Antibiotics kill bacteria.",
            "Bacteria are killed by antibiotics.",
            "Antibiotics are effective against bacteria.",
        ],
        negative_phrasings=[
            "Antibiotics do not kill bacteria.",
            "Bacteria are not killed by antibiotics.",
            "Antibiotics are not effective against bacteria.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Antibiotic function",
    ),
    ClaimSet(
        positive_phrasings=[
            "The heart pumps blood through the body.",
            "Blood is pumped through the body by the heart.",
            "The heart's function is to pump blood.",
        ],
        negative_phrasings=[
            "The heart does not pump blood through the body.",
            "Blood is not pumped through the body by the heart.",
            "The heart's function is not to pump blood.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Heart function",
    ),
    ClaimSet(
        positive_phrasings=[
            "Plants perform photosynthesis.",
            "Photosynthesis is performed by plants.",
            "Plants convert light energy through photosynthesis.",
        ],
        negative_phrasings=[
            "Plants do not perform photosynthesis.",
            "Photosynthesis is not performed by plants.",
            "Plants do not convert light energy through photosynthesis.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Plant photosynthesis",
    ),
    ClaimSet(
        positive_phrasings=[
            "The Moon orbits the Earth.",
            "Earth is orbited by the Moon.",
            "The Moon revolves around Earth.",
        ],
        negative_phrasings=[
            "The Moon does not orbit the Earth.",
            "Earth is not orbited by the Moon.",
            "The Moon does not revolve around Earth.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Moon's orbit",
    ),
    ClaimSet(
        positive_phrasings=[
            "Diamonds are made of carbon.",
            "Carbon is the primary element in diamonds.",
            "Diamonds consist of carbon atoms.",
        ],
        negative_phrasings=[
            "Diamonds are not made of carbon.",
            "Carbon is not the primary element in diamonds.",
            "Diamonds do not consist of carbon atoms.",
        ],
        belief_type=BeliefType.WELL_ESTABLISHED_FACT,
        description="Diamond composition",
    ),
]

CONTESTED_FACTS = [
    ClaimSet(
        positive_phrasings=[
            "Human activity is the primary cause of recent global warming.",
            "Recent global warming is primarily caused by human activity.",
            "The main driver of contemporary climate change is human activity.",
        ],
        negative_phrasings=[
            "Human activity is not the primary cause of recent global warming.",
            "Recent global warming is not primarily caused by human activity.",
            "The main driver of contemporary climate change is not human activity.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Climate change attribution",
    ),
    ClaimSet(
        positive_phrasings=[
            "COVID-19 vaccines are effective at preventing severe illness.",
            "Vaccination against COVID-19 effectively prevents severe disease.",
            "COVID-19 vaccines work to prevent serious illness.",
        ],
        negative_phrasings=[
            "COVID-19 vaccines are not effective at preventing severe illness.",
            "Vaccination against COVID-19 does not effectively prevent severe disease.",
            "COVID-19 vaccines do not work to prevent serious illness.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Vaccine efficacy",
    ),
    ClaimSet(
        positive_phrasings=[
            "Minimum wage increases lead to reduced employment.",
            "Raising the minimum wage causes job losses.",
            "Higher minimum wages result in fewer jobs.",
        ],
        negative_phrasings=[
            "Minimum wage increases do not lead to reduced employment.",
            "Raising the minimum wage does not cause job losses.",
            "Higher minimum wages do not result in fewer jobs.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Minimum wage effects",
    ),
    ClaimSet(
        positive_phrasings=[
            "Genetically modified foods are safe for human consumption.",
            "GMO foods pose no health risks to humans.",
            "Consuming genetically modified organisms is safe.",
        ],
        negative_phrasings=[
            "Genetically modified foods are not safe for human consumption.",
            "GMO foods pose health risks to humans.",
            "Consuming genetically modified organisms is not safe.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="GMO safety",
    ),
    ClaimSet(
        positive_phrasings=[
            "Nuclear energy is safer than fossil fuels.",
            "Fossil fuels are more dangerous than nuclear energy.",
            "Nuclear power poses fewer risks than fossil fuel energy.",
        ],
        negative_phrasings=[
            "Nuclear energy is not safer than fossil fuels.",
            "Fossil fuels are not more dangerous than nuclear energy.",
            "Nuclear power does not pose fewer risks than fossil fuel energy.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Nuclear vs fossil fuel safety",
    ),
    ClaimSet(
        positive_phrasings=[
            "Gun control reduces violent crime.",
            "Stricter gun laws decrease violent crime rates.",
            "Violent crime is reduced by gun control measures.",
        ],
        negative_phrasings=[
            "Gun control does not reduce violent crime.",
            "Stricter gun laws do not decrease violent crime rates.",
            "Violent crime is not reduced by gun control measures.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Gun control effectiveness",
    ),
    ClaimSet(
        positive_phrasings=[
            "Tax cuts stimulate economic growth.",
            "Economic growth is stimulated by tax cuts.",
            "Reducing taxes leads to economic expansion.",
        ],
        negative_phrasings=[
            "Tax cuts do not stimulate economic growth.",
            "Economic growth is not stimulated by tax cuts.",
            "Reducing taxes does not lead to economic expansion.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Tax policy effects",
    ),
    ClaimSet(
        positive_phrasings=[
            "School choice improves educational outcomes.",
            "Educational outcomes are improved by school choice programs.",
            "Student achievement increases with school choice.",
        ],
        negative_phrasings=[
            "School choice does not improve educational outcomes.",
            "Educational outcomes are not improved by school choice programs.",
            "Student achievement does not increase with school choice.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="School choice effects",
    ),
    ClaimSet(
        positive_phrasings=[
            "Organic food is healthier than conventional food.",
            "Conventional food is less healthy than organic food.",
            "Organic produce provides greater health benefits.",
        ],
        negative_phrasings=[
            "Organic food is not healthier than conventional food.",
            "Conventional food is not less healthy than organic food.",
            "Organic produce does not provide greater health benefits.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Organic food health benefits",
    ),
    ClaimSet(
        positive_phrasings=[
            "Social media use causes mental health problems in teenagers.",
            "Teenage mental health problems are caused by social media.",
            "Social media negatively impacts adolescent mental health.",
        ],
        negative_phrasings=[
            "Social media use does not cause mental health problems in teenagers.",
            "Teenage mental health problems are not caused by social media.",
            "Social media does not negatively impact adolescent mental health.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Social media and mental health",
    ),
    ClaimSet(
        positive_phrasings=[
            "Violent video games increase aggressive behavior.",
            "Aggressive behavior is increased by violent video games.",
            "Playing violent games makes people more aggressive.",
        ],
        negative_phrasings=[
            "Violent video games do not increase aggressive behavior.",
            "Aggressive behavior is not increased by violent video games.",
            "Playing violent games does not make people more aggressive.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Video game violence effects",
    ),
    ClaimSet(
        positive_phrasings=[
            "Immigration benefits the economy.",
            "The economy benefits from immigration.",
            "Immigration provides net economic gains.",
        ],
        negative_phrasings=[
            "Immigration does not benefit the economy.",
            "The economy does not benefit from immigration.",
            "Immigration does not provide net economic gains.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Immigration economic impact",
    ),
    ClaimSet(
        positive_phrasings=[
            "Capital punishment deters crime.",
            "Crime is deterred by capital punishment.",
            "The death penalty reduces criminal activity.",
        ],
        negative_phrasings=[
            "Capital punishment does not deter crime.",
            "Crime is not deterred by capital punishment.",
            "The death penalty does not reduce criminal activity.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Death penalty deterrence",
    ),
    ClaimSet(
        positive_phrasings=[
            "Artificial sweeteners are harmful to health.",
            "Health is harmed by artificial sweeteners.",
            "Using artificial sweeteners poses health risks.",
        ],
        negative_phrasings=[
            "Artificial sweeteners are not harmful to health.",
            "Health is not harmed by artificial sweeteners.",
            "Using artificial sweeteners does not pose health risks.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Artificial sweetener safety",
    ),
    ClaimSet(
        positive_phrasings=[
            "Remote work increases productivity.",
            "Productivity is increased by remote work.",
            "Working from home makes employees more productive.",
        ],
        negative_phrasings=[
            "Remote work does not increase productivity.",
            "Productivity is not increased by remote work.",
            "Working from home does not make employees more productive.",
        ],
        belief_type=BeliefType.CONTESTED_FACT,
        description="Remote work productivity",
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
