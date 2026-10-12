"""Question battery over data/sample_texts/.

Question types:
  • factual    — specific facts extractable from a single passage
  • relational — questions requiring entity relationships
  • thematic   — broad synthesis across the whole corpus
  • multi_hop  — connecting concepts across documents or chains of events

`expected_entities` drive the string-match metrics (metrics.py).
`reference_answer` is what the LLM judge grades correctness against; every
fact in it is stated in the corpus. For thematic questions it lists key points
rather than one canonical answer.
"""

DATASET = "sample-texts-v1"

QUESTIONS = [
    # Factual — single-passage lookup
    {
        "id": "f1",
        "type": "factual",
        "question": "When was the Dartmouth Conference held and who organized it?",
        "expected_entities": ["Dartmouth Conference", "John McCarthy", "Marvin Minsky", "1956"],
        "reference_answer": (
            "The Dartmouth Conference was held in 1956 and formally founded the field of AI. "
            "It was organized by John McCarthy (who coined the term 'artificial intelligence') "
            "together with Marvin Minsky, Nathaniel Rochester, and Claude Shannon."
        ),
    },
    {
        "id": "f2",
        "type": "factual",
        "question": "What is the Paris Agreement and when was it adopted?",
        "expected_entities": ["Paris Agreement", "2015", "UNFCCC", "2°C"],
        "reference_answer": (
            "The Paris Agreement is a landmark international climate accord adopted in 2015 under the "
            "United Nations Framework Convention on Climate Change (UNFCCC). Countries committed to "
            "limiting warming to well below 2°C above pre-industrial levels, with efforts toward 1.5°C. "
            "Current national pledges (NDCs) are insufficient to meet these targets."
        ),
    },
    {
        "id": "f3",
        "type": "factual",
        "question": "What happened to Julius Caesar on the Ides of March?",
        "expected_entities": ["Julius Caesar", "Ides of March", "Brutus", "Cassius", "44 BC"],
        "reference_answer": (
            "On the Ides of March (March 15) 44 BC, Caesar was stabbed 23 times in the Theatre of Pompey "
            "by a conspiracy of about 60 senators (the Liberatores) led by Marcus Junius Brutus and "
            "Gaius Cassius Longinus. His death plunged Rome into another round of civil wars."
        ),
    },

    # Relational — entity relationships
    {
        "id": "r1",
        "type": "relational",
        "question": "How were Julius Caesar, Pompey, and Crassus connected, and what happened to their alliance?",
        "expected_entities": ["Julius Caesar", "Pompey", "Crassus", "First Triumvirate"],
        "reference_answer": (
            "In 60 BC they formed the First Triumvirate, an informal political alliance that dominated "
            "Roman politics: Pompey brought military fame, Crassus wealth, and Caesar political "
            "maneuvering. It collapsed after Crassus was killed at the Battle of Carrhae in 53 BC "
            "fighting the Parthians; Caesar and Pompey became rivals, leading to civil war after Caesar "
            "crossed the Rubicon in 49 BC and his defeat of Pompey at Pharsalus in 48 BC."
        ),
    },
    {
        "id": "r2",
        "type": "relational",
        "question": "What is the relationship between Geoffrey Hinton, Yann LeCun, and Yoshua Bengio in the history of AI?",
        "expected_entities": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "deep learning"],
        "reference_answer": (
            "They are often called the 'Godfathers of AI': all three persisted with neural network research "
            "through the AI winters. Hinton worked on backpropagation and deep learning (his Toronto group "
            "built AlexNet, which won ImageNet 2012), LeCun developed convolutional neural networks for image "
            "recognition, and Bengio contributed foundational work on recurrent networks and attention."
        ),
    },
    {
        "id": "r3",
        "type": "relational",
        "question": "How does the Keeling Curve relate to the IPCC and climate policy?",
        "expected_entities": ["Keeling", "IPCC", "CO2", "climate"],
        "reference_answer": (
            "The Keeling Curve, from Charles Keeling's continuous CO2 measurements at Mauna Loa since 1958, "
            "documents the steady rise of atmospheric CO2 from 315 ppm to over 420 ppm. This kind of evidence "
            "underpins the IPCC (est. 1988 by UNEP and WMO), whose assessment reports (e.g. AR6) synthesize "
            "climate science and confirm human influence; that science informs policy such as the Paris "
            "Agreement. The corpus does not state an explicit direct link between the curve and the IPCC."
        ),
    },

    # Thematic — broad synthesis
    {
        "id": "t1",
        "type": "thematic",
        "question": "What overarching themes connect the stories of Julius Caesar, the AI researchers, and climate scientists?",
        "expected_entities": [],
        "reference_answer": (
            "Key points: (1) individuals and alliances driving transformative change (Caesar and the "
            "triumvirates; McCarthy, Minsky, Hinton/LeCun/Bengio; Arrhenius, Keeling, Hansen); "
            "(2) persistence through setbacks (AI winters; long-term CO2 measurement; Caesar's rise); "
            "(3) concentration of power or influence and the institutional backlash it provokes (Senate vs "
            "Caesar; AI safety concerns; politicization and harassment of climate scientists); "
            "(4) transitions between eras (Republic to Empire; symbolic AI to deep learning; fossil fuels "
            "to renewables)."
        ),
    },
    {
        "id": "t2",
        "type": "thematic",
        "question": "How have major paradigm shifts occurred in the three domains covered by these documents?",
        "expected_entities": [],
        "reference_answer": (
            "Key points: Rome shifted from Republic to Empire through Caesar's civil war, his assassination, "
            "and Octavian becoming Augustus in 27 BC while keeping republican appearances. AI shifted from "
            "symbolic AI and expert systems, through two AI winters, to deep learning (AlexNet 2012), "
            "transformers (2017), and LLMs. Climate science moved from early calculations (Arrhenius 1896) "
            "and measurements (Keeling 1958) to scientific consensus (IPCC) and policy (Paris 2015), with an "
            "energy transition toward renewables. In each case accumulated evidence or pressure built up "
            "before a decisive break."
        ),
    },

    # Multi-hop — chains of events
    {
        "id": "m1",
        "type": "multi_hop",
        "question": "Trace how the collapse of the First Triumvirate led step by step to the Roman Empire under Augustus.",
        "expected_entities": ["First Triumvirate", "Crassus", "Pompey", "civil war", "Octavian", "Augustus"],
        "reference_answer": (
            "Crassus died at Carrhae (53 BC), ending the triumvirate; Caesar and Pompey became rivals. The "
            "Senate ordered Caesar to disband his army; he crossed the Rubicon (49 BC), starting civil war, and "
            "defeated Pompey at Pharsalus (48 BC). Caesar became dictator perpetuo (44 BC) and was assassinated "
            "on the Ides of March. Octavian, Antony, and Lepidus formed the Second Triumvirate and defeated "
            "Brutus and Cassius at Philippi (42 BC). The Second Triumvirate fractured; Octavian defeated Antony "
            "and Cleopatra at Actium (31 BC). In 27 BC Octavian received the title Augustus, marking the "
            "transition from Republic to Empire."
        ),
    },
    {
        "id": "m2",
        "type": "multi_hop",
        "question": "How did the transition from symbolic AI to deep learning happen, and who were the key figures at each stage?",
        "expected_entities": ["symbolic AI", "expert systems", "neural network", "Hinton", "deep learning"],
        "reference_answer": (
            "Symbolic AI was pioneered by Newell and Simon (Logic Theorist, General Problem Solver) and McCarthy "
            "(LISP), after the 1956 Dartmouth Conference. The first AI winter hit in the 1970s (Lighthill Report). "
            "Expert systems (DENDRAL, MYCIN) revived the field in the 1980s but proved brittle, causing a second "
            "winter in the late 1980s-early 1990s. Hinton, LeCun, and Bengio persisted with neural networks "
            "(backpropagation, CNNs, RNNs/attention). AlexNet from Hinton's group won ImageNet 2012, launching the "
            "deep learning era; the 2017 transformer paper (Vaswani et al.) then enabled modern LLMs."
        ),
    },
]
