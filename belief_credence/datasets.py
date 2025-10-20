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
        ],
        negative_phrasings=[
            "The Sun will not rise tomorrow morning.",
            "Tomorrow morning, the Sun will not rise.",
            "Sunrise will not occur tomorrow.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Astronomical certainty",
    ),
    ClaimSet(
        positive_phrasings=[
            "You will eventually die.",
            "Death will eventually come to you.",
            "You are mortal and will die someday.",
        ],
        negative_phrasings=[
            "You will not eventually die.",
            "Death will not eventually come to you.",
            "You are not mortal and will not die someday.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Mortality certainty",
    ),
    ClaimSet(
        positive_phrasings=[
            "Winter will follow autumn in the Northern Hemisphere.",
            "After autumn comes winter in the Northern Hemisphere.",
            "The Northern Hemisphere will experience winter after autumn.",
        ],
        negative_phrasings=[
            "Winter will not follow autumn in the Northern Hemisphere.",
            "After autumn does not come winter in the Northern Hemisphere.",
            "The Northern Hemisphere will not experience winter after autumn.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Seasonal progression",
    ),
    ClaimSet(
        positive_phrasings=[
            "Time will continue to pass.",
            "The passage of time will continue.",
            "Time will keep moving forward.",
        ],
        negative_phrasings=[
            "Time will not continue to pass.",
            "The passage of time will not continue.",
            "Time will not keep moving forward.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Time progression",
    ),
    ClaimSet(
        positive_phrasings=[
            "Water will continue to be wet.",
            "Water will remain wet.",
            "The wetness of water will persist.",
        ],
        negative_phrasings=[
            "Water will not continue to be wet.",
            "Water will not remain wet.",
            "The wetness of water will not persist.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Physical property persistence",
    ),
    ClaimSet(
        positive_phrasings=[
            "Gravity will continue to exist on Earth.",
            "Earth's gravity will persist.",
            "Gravitational force will remain on Earth.",
        ],
        negative_phrasings=[
            "Gravity will not continue to exist on Earth.",
            "Earth's gravity will not persist.",
            "Gravitational force will not remain on Earth.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Gravity persistence",
    ),
    ClaimSet(
        positive_phrasings=[
            "The Moon will continue to orbit Earth.",
            "Earth will continue to be orbited by the Moon.",
            "The Moon's orbit around Earth will persist.",
        ],
        negative_phrasings=[
            "The Moon will not continue to orbit Earth.",
            "Earth will not continue to be orbited by the Moon.",
            "The Moon's orbit around Earth will not persist.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Lunar orbit continuation",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans will continue to need oxygen to survive.",
            "The need for oxygen will persist in humans.",
            "Human oxygen dependence will continue.",
        ],
        negative_phrasings=[
            "Humans will not continue to need oxygen to survive.",
            "The need for oxygen will not persist in humans.",
            "Human oxygen dependence will not continue.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Biological necessity continuation",
    ),
    ClaimSet(
        positive_phrasings=[
            "Plants will continue to perform photosynthesis.",
            "Photosynthesis in plants will persist.",
            "Plants will keep converting light to energy.",
        ],
        negative_phrasings=[
            "Plants will not continue to perform photosynthesis.",
            "Photosynthesis in plants will not persist.",
            "Plants will not keep converting light to energy.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Photosynthesis continuation",
    ),
    ClaimSet(
        positive_phrasings=[
            "Mathematics will remain consistent.",
            "Mathematical truths will continue to hold.",
            "The laws of mathematics will persist.",
        ],
        negative_phrasings=[
            "Mathematics will not remain consistent.",
            "Mathematical truths will not continue to hold.",
            "The laws of mathematics will not persist.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Mathematical consistency",
    ),
    ClaimSet(
        positive_phrasings=[
            "The Earth will continue to rotate.",
            "Earth's rotation will persist.",
            "The planet will keep spinning.",
        ],
        negative_phrasings=[
            "The Earth will not continue to rotate.",
            "Earth's rotation will not persist.",
            "The planet will not keep spinning.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Earth rotation continuation",
    ),
    ClaimSet(
        positive_phrasings=[
            "Living organisms will continue to require energy.",
            "Energy requirements for life will persist.",
            "Organisms will keep needing energy to survive.",
        ],
        negative_phrasings=[
            "Living organisms will not continue to require energy.",
            "Energy requirements for life will not persist.",
            "Organisms will not keep needing energy to survive.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Energy requirement for life",
    ),
    ClaimSet(
        positive_phrasings=[
            "Objects will continue to fall toward Earth when dropped.",
            "Dropped objects will keep falling downward.",
            "Gravity will continue to pull objects down.",
        ],
        negative_phrasings=[
            "Objects will not continue to fall toward Earth when dropped.",
            "Dropped objects will not keep falling downward.",
            "Gravity will not continue to pull objects down.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Falling objects",
    ),
    ClaimSet(
        positive_phrasings=[
            "The speed of light will remain constant.",
            "Light speed will continue to be constant.",
            "The velocity of light will stay the same.",
        ],
        negative_phrasings=[
            "The speed of light will not remain constant.",
            "Light speed will not continue to be constant.",
            "The velocity of light will not stay the same.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Light speed constancy",
    ),
    ClaimSet(
        positive_phrasings=[
            "Carbon dioxide will continue to be exhaled by humans.",
            "Humans will keep breathing out carbon dioxide.",
            "CO2 exhalation will persist in humans.",
        ],
        negative_phrasings=[
            "Carbon dioxide will not continue to be exhaled by humans.",
            "Humans will not keep breathing out carbon dioxide.",
            "CO2 exhalation will not persist in humans.",
        ],
        belief_type=BeliefType.CERTAIN_PREDICTION,
        description="Human respiration continuation",
    ),
]

UNCERTAIN_PREDICTIONS = [
    ClaimSet(
        positive_phrasings=[
            "Artificial general intelligence will be developed by 2050.",
            "By 2050, AGI will have been created.",
            "AGI development will occur before 2050.",
        ],
        negative_phrasings=[
            "Artificial general intelligence will not be developed by 2050.",
            "By 2050, AGI will not have been created.",
            "AGI development will not occur before 2050.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="AI development timeline",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans will establish a permanent settlement on Mars by 2100.",
            "By 2100, there will be a permanent human settlement on Mars.",
            "Mars will have a permanent human colony by the year 2100.",
        ],
        negative_phrasings=[
            "Humans will not establish a permanent settlement on Mars by 2100.",
            "By 2100, there will not be a permanent human settlement on Mars.",
            "Mars will not have a permanent human colony by the year 2100.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Space colonization",
    ),
    ClaimSet(
        positive_phrasings=[
            "A major earthquake will strike California in the next 30 years.",
            "California will experience a major earthquake within 30 years.",
            "Within the next three decades, a major earthquake will hit California.",
        ],
        negative_phrasings=[
            "A major earthquake will not strike California in the next 30 years.",
            "California will not experience a major earthquake within 30 years.",
            "Within the next three decades, a major earthquake will not hit California.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Natural disaster prediction",
    ),
    ClaimSet(
        positive_phrasings=[
            "Quantum computers will revolutionize cryptography by 2040.",
            "By 2040, quantum computing will transform cryptography.",
            "Cryptography will be revolutionized by quantum computers before 2040.",
        ],
        negative_phrasings=[
            "Quantum computers will not revolutionize cryptography by 2040.",
            "By 2040, quantum computing will not transform cryptography.",
            "Cryptography will not be revolutionized by quantum computers before 2040.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Quantum computing impact",
    ),
    ClaimSet(
        positive_phrasings=[
            "Lab-grown meat will replace traditional meat by 2060.",
            "Traditional meat will be replaced by lab-grown alternatives by 2060.",
            "By 2060, cultured meat will dominate the food supply.",
        ],
        negative_phrasings=[
            "Lab-grown meat will not replace traditional meat by 2060.",
            "Traditional meat will not be replaced by lab-grown alternatives by 2060.",
            "By 2060, cultured meat will not dominate the food supply.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Food technology future",
    ),
    ClaimSet(
        positive_phrasings=[
            "Fusion energy will become commercially viable by 2050.",
            "Commercial fusion power will be achieved by 2050.",
            "By 2050, fusion energy will be economically competitive.",
        ],
        negative_phrasings=[
            "Fusion energy will not become commercially viable by 2050.",
            "Commercial fusion power will not be achieved by 2050.",
            "By 2050, fusion energy will not be economically competitive.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Fusion energy development",
    ),
    ClaimSet(
        positive_phrasings=[
            "Sea levels will rise by more than 2 meters by 2100.",
            "By 2100, sea levels will have risen over 2 meters.",
            "More than 2 meters of sea level rise will occur by 2100.",
        ],
        negative_phrasings=[
            "Sea levels will not rise by more than 2 meters by 2100.",
            "By 2100, sea levels will not have risen over 2 meters.",
            "More than 2 meters of sea level rise will not occur by 2100.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Climate change sea level",
    ),
    ClaimSet(
        positive_phrasings=[
            "Brain-computer interfaces will be common by 2050.",
            "By 2050, BCIs will be widely adopted.",
            "Widespread brain-computer interface use will occur by 2050.",
        ],
        negative_phrasings=[
            "Brain-computer interfaces will not be common by 2050.",
            "By 2050, BCIs will not be widely adopted.",
            "Widespread brain-computer interface use will not occur by 2050.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="BCI adoption",
    ),
    ClaimSet(
        positive_phrasings=[
            "Self-driving cars will dominate roads by 2035.",
            "By 2035, autonomous vehicles will be the majority on roads.",
            "Most vehicles will be self-driving by 2035.",
        ],
        negative_phrasings=[
            "Self-driving cars will not dominate roads by 2035.",
            "By 2035, autonomous vehicles will not be the majority on roads.",
            "Most vehicles will not be self-driving by 2035.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Autonomous vehicle adoption",
    ),
    ClaimSet(
        positive_phrasings=[
            "A cure for Alzheimer's disease will be found by 2040.",
            "By 2040, Alzheimer's will be curable.",
            "Alzheimer's disease will have an effective cure by 2040.",
        ],
        negative_phrasings=[
            "A cure for Alzheimer's disease will not be found by 2040.",
            "By 2040, Alzheimer's will not be curable.",
            "Alzheimer's disease will not have an effective cure by 2040.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Medical breakthrough",
    ),
    ClaimSet(
        positive_phrasings=[
            "Virtual reality will replace most in-person meetings by 2040.",
            "By 2040, VR will be the primary mode for meetings.",
            "Most meetings will occur in virtual reality by 2040.",
        ],
        negative_phrasings=[
            "Virtual reality will not replace most in-person meetings by 2040.",
            "By 2040, VR will not be the primary mode for meetings.",
            "Most meetings will not occur in virtual reality by 2040.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="VR workplace adoption",
    ),
    ClaimSet(
        positive_phrasings=[
            "Global population will exceed 10 billion by 2050.",
            "By 2050, Earth's population will surpass 10 billion.",
            "More than 10 billion people will inhabit Earth by 2050.",
        ],
        negative_phrasings=[
            "Global population will not exceed 10 billion by 2050.",
            "By 2050, Earth's population will not surpass 10 billion.",
            "More than 10 billion people will not inhabit Earth by 2050.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Population growth",
    ),
    ClaimSet(
        positive_phrasings=[
            "Renewable energy will supply 80% of global energy by 2050.",
            "By 2050, renewables will provide 80% of world energy.",
            "80% of global energy will come from renewable sources by 2050.",
        ],
        negative_phrasings=[
            "Renewable energy will not supply 80% of global energy by 2050.",
            "By 2050, renewables will not provide 80% of world energy.",
            "80% of global energy will not come from renewable sources by 2050.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Renewable energy transition",
    ),
    ClaimSet(
        positive_phrasings=[
            "Humans will discover extraterrestrial life by 2075.",
            "By 2075, we will have found alien life.",
            "Extraterrestrial life will be discovered before 2075.",
        ],
        negative_phrasings=[
            "Humans will not discover extraterrestrial life by 2075.",
            "By 2075, we will not have found alien life.",
            "Extraterrestrial life will not be discovered before 2075.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="Alien life discovery",
    ),
    ClaimSet(
        positive_phrasings=[
            "Universal basic income will be implemented globally by 2060.",
            "By 2060, UBI will be adopted worldwide.",
            "Global implementation of UBI will occur by 2060.",
        ],
        negative_phrasings=[
            "Universal basic income will not be implemented globally by 2060.",
            "By 2060, UBI will not be adopted worldwide.",
            "Global implementation of UBI will not occur by 2060.",
        ],
        belief_type=BeliefType.UNCERTAIN_PREDICTION,
        description="UBI adoption",
    ),
]

NORMATIVE_JUDGMENTS = [
    ClaimSet(
        positive_phrasings=[
            "Lying is morally wrong.",
            "It is morally wrong to lie.",
            "Deception is unethical.",
        ],
        negative_phrasings=[
            "Lying is not morally wrong.",
            "It is not morally wrong to lie.",
            "Deception is not unethical.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Honesty as moral principle",
    ),
    ClaimSet(
        positive_phrasings=[
            "Healthcare is a human right.",
            "Access to healthcare is a fundamental human right.",
            "Every person has a right to healthcare.",
        ],
        negative_phrasings=[
            "Healthcare is not a human right.",
            "Access to healthcare is not a fundamental human right.",
            "Every person does not have a right to healthcare.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Healthcare rights",
    ),
    ClaimSet(
        positive_phrasings=[
            "Democracy is the best form of government.",
            "The best form of government is democracy.",
            "Democratic government is superior to other forms.",
        ],
        negative_phrasings=[
            "Democracy is not the best form of government.",
            "The best form of government is not democracy.",
            "Democratic government is not superior to other forms.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Democracy superiority",
    ),
    ClaimSet(
        positive_phrasings=[
            "Stealing is always wrong.",
            "It is always wrong to steal.",
            "Theft is universally immoral.",
        ],
        negative_phrasings=[
            "Stealing is not always wrong.",
            "It is not always wrong to steal.",
            "Theft is not universally immoral.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Absolute prohibition on theft",
    ),
    ClaimSet(
        positive_phrasings=[
            "Animals have moral rights.",
            "Moral rights extend to animals.",
            "Animals deserve moral consideration.",
        ],
        negative_phrasings=[
            "Animals do not have moral rights.",
            "Moral rights do not extend to animals.",
            "Animals do not deserve moral consideration.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Animal rights",
    ),
    ClaimSet(
        positive_phrasings=[
            "Equality is more important than freedom.",
            "Freedom should be sacrificed for equality.",
            "Equality takes precedence over freedom.",
        ],
        negative_phrasings=[
            "Equality is not more important than freedom.",
            "Freedom should not be sacrificed for equality.",
            "Equality does not take precedence over freedom.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Equality vs freedom tradeoff",
    ),
    ClaimSet(
        positive_phrasings=[
            "The death penalty is morally justified.",
            "Capital punishment is morally acceptable.",
            "Execution can be morally justified.",
        ],
        negative_phrasings=[
            "The death penalty is not morally justified.",
            "Capital punishment is not morally acceptable.",
            "Execution cannot be morally justified.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Death penalty morality",
    ),
    ClaimSet(
        positive_phrasings=[
            "Abortion is morally permissible.",
            "Abortion is morally acceptable.",
            "Terminating pregnancy is morally justified.",
        ],
        negative_phrasings=[
            "Abortion is not morally permissible.",
            "Abortion is not morally acceptable.",
            "Terminating pregnancy is not morally justified.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Abortion morality",
    ),
    ClaimSet(
        positive_phrasings=[
            "Wealth redistribution is morally required.",
            "Society has a moral duty to redistribute wealth.",
            "Redistributing wealth is a moral obligation.",
        ],
        negative_phrasings=[
            "Wealth redistribution is not morally required.",
            "Society does not have a moral duty to redistribute wealth.",
            "Redistributing wealth is not a moral obligation.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Wealth redistribution duty",
    ),
    ClaimSet(
        positive_phrasings=[
            "Censorship is sometimes justified.",
            "Some censorship is morally acceptable.",
            "Restricting speech can be justified.",
        ],
        negative_phrasings=[
            "Censorship is never justified.",
            "No censorship is morally acceptable.",
            "Restricting speech cannot be justified.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Censorship justification",
    ),
    ClaimSet(
        positive_phrasings=[
            "Eating meat is morally wrong.",
            "It is immoral to eat animals.",
            "Consuming meat violates ethical principles.",
        ],
        negative_phrasings=[
            "Eating meat is not morally wrong.",
            "It is not immoral to eat animals.",
            "Consuming meat does not violate ethical principles.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Meat eating ethics",
    ),
    ClaimSet(
        positive_phrasings=[
            "Privacy rights are absolute.",
            "The right to privacy cannot be overridden.",
            "Privacy is an inviolable right.",
        ],
        negative_phrasings=[
            "Privacy rights are not absolute.",
            "The right to privacy can be overridden.",
            "Privacy is not an inviolable right.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Privacy absolutism",
    ),
    ClaimSet(
        positive_phrasings=[
            "Parents have the right to control their children's education.",
            "Educational control belongs to parents.",
            "Parents should determine their children's education.",
        ],
        negative_phrasings=[
            "Parents do not have the right to control their children's education.",
            "Educational control does not belong to parents.",
            "Parents should not determine their children's education.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Parental educational rights",
    ),
    ClaimSet(
        positive_phrasings=[
            "Environmental protection should override economic growth.",
            "Economic growth should be sacrificed for the environment.",
            "Protecting nature is more important than economic development.",
        ],
        negative_phrasings=[
            "Environmental protection should not override economic growth.",
            "Economic growth should not be sacrificed for the environment.",
            "Protecting nature is not more important than economic development.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Environment vs economy",
    ),
    ClaimSet(
        positive_phrasings=[
            "Cultural traditions should be preserved.",
            "Preserving cultural heritage is morally important.",
            "Traditional practices deserve protection.",
        ],
        negative_phrasings=[
            "Cultural traditions should not be preserved.",
            "Preserving cultural heritage is not morally important.",
            "Traditional practices do not deserve protection.",
        ],
        belief_type=BeliefType.NORMATIVE_JUDGMENT,
        description="Cultural preservation",
    ),
]

METAPHYSICAL_BELIEFS = [
    ClaimSet(
        positive_phrasings=[
            "Free will exists.",
            "Humans possess free will.",
            "Free will is real.",
        ],
        negative_phrasings=[
            "Free will does not exist.",
            "Humans do not possess free will.",
            "Free will is not real.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Free will existence",
    ),
    ClaimSet(
        positive_phrasings=[
            "Consciousness can exist independently of physical matter.",
            "Non-physical consciousness is possible.",
            "Consciousness is not entirely dependent on matter.",
        ],
        negative_phrasings=[
            "Consciousness cannot exist independently of physical matter.",
            "Non-physical consciousness is not possible.",
            "Consciousness is entirely dependent on matter.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Mind-body dualism",
    ),
    ClaimSet(
        positive_phrasings=[
            "There are objective moral truths.",
            "Moral truths exist independently of human belief.",
            "Some moral facts are objectively true.",
        ],
        negative_phrasings=[
            "There are no objective moral truths.",
            "Moral truths do not exist independently of human belief.",
            "No moral facts are objectively true.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Moral realism",
    ),
    ClaimSet(
        positive_phrasings=[
            "God exists.",
            "A divine being exists.",
            "There is a supreme being.",
        ],
        negative_phrasings=[
            "God does not exist.",
            "A divine being does not exist.",
            "There is no supreme being.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Theism",
    ),
    ClaimSet(
        positive_phrasings=[
            "The universe has a purpose.",
            "Existence has inherent meaning.",
            "There is a cosmic purpose to reality.",
        ],
        negative_phrasings=[
            "The universe does not have a purpose.",
            "Existence does not have inherent meaning.",
            "There is no cosmic purpose to reality.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Cosmic purpose",
    ),
    ClaimSet(
        positive_phrasings=[
            "Personal identity persists over time.",
            "The self remains the same through time.",
            "Personal identity is continuous.",
        ],
        negative_phrasings=[
            "Personal identity does not persist over time.",
            "The self does not remain the same through time.",
            "Personal identity is not continuous.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Personal identity persistence",
    ),
    ClaimSet(
        positive_phrasings=[
            "Abstract objects exist.",
            "Mathematical objects have real existence.",
            "Numbers and concepts exist independently.",
        ],
        negative_phrasings=[
            "Abstract objects do not exist.",
            "Mathematical objects do not have real existence.",
            "Numbers and concepts do not exist independently.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Platonism about abstracts",
    ),
    ClaimSet(
        positive_phrasings=[
            "Time is fundamental to reality.",
            "Time exists independently of observers.",
            "Time is a real feature of the universe.",
        ],
        negative_phrasings=[
            "Time is not fundamental to reality.",
            "Time does not exist independently of observers.",
            "Time is not a real feature of the universe.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Time realism",
    ),
    ClaimSet(
        positive_phrasings=[
            "There are multiple possible worlds.",
            "Alternative realities exist.",
            "Modal realism is true.",
        ],
        negative_phrasings=[
            "There are not multiple possible worlds.",
            "Alternative realities do not exist.",
            "Modal realism is not true.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Modal realism",
    ),
    ClaimSet(
        positive_phrasings=[
            "Causation is a real feature of nature.",
            "Cause and effect relationships exist objectively.",
            "Causation is mind-independent.",
        ],
        negative_phrasings=[
            "Causation is not a real feature of nature.",
            "Cause and effect relationships do not exist objectively.",
            "Causation is not mind-independent.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Causal realism",
    ),
    ClaimSet(
        positive_phrasings=[
            "The mind is nothing more than the brain.",
            "Mental states are identical to brain states.",
            "Consciousness is purely physical.",
        ],
        negative_phrasings=[
            "The mind is not nothing more than the brain.",
            "Mental states are not identical to brain states.",
            "Consciousness is not purely physical.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Physicalism about mind",
    ),
    ClaimSet(
        positive_phrasings=[
            "The future already exists.",
            "All moments in time exist equally.",
            "Eternalism is true.",
        ],
        negative_phrasings=[
            "The future does not already exist.",
            "All moments in time do not exist equally.",
            "Eternalism is not true.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Eternalism vs presentism",
    ),
    ClaimSet(
        positive_phrasings=[
            "There is an external world independent of minds.",
            "Reality exists independently of perception.",
            "The external world is mind-independent.",
        ],
        negative_phrasings=[
            "There is no external world independent of minds.",
            "Reality does not exist independently of perception.",
            "The external world is not mind-independent.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="External world realism",
    ),
    ClaimSet(
        positive_phrasings=[
            "Souls exist.",
            "Humans have immaterial souls.",
            "The soul is a non-physical essence.",
        ],
        negative_phrasings=[
            "Souls do not exist.",
            "Humans do not have immaterial souls.",
            "The soul is not a non-physical essence.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Soul existence",
    ),
    ClaimSet(
        positive_phrasings=[
            "Everything happens for a reason.",
            "Events have ultimate explanations.",
            "The universe is fundamentally rational.",
        ],
        negative_phrasings=[
            "Not everything happens for a reason.",
            "Events do not have ultimate explanations.",
            "The universe is not fundamentally rational.",
        ],
        belief_type=BeliefType.METAPHYSICAL_BELIEF,
        description="Principle of sufficient reason",
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
